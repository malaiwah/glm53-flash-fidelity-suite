#!/usr/bin/env python3
"""Focused zero-cost lifecycle tests.  No provider mutation or network access."""
import base64
import calendar
import json
import stat
from dataclasses import replace
from decimal import Decimal
import email.message
import io
import hashlib
import multiprocessing
import os
import signal
import subprocess
import sys
import shutil
import tempfile
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

import fidelity.cloudlease as cloudlease_module  # noqa: E402
import fidelity.runpodapi as runpod_module  # noqa: E402
from fidelity.campaign import CampaignLedger, attempt_key  # noqa: E402
from fidelity.cloudlease import (  # noqa: E402
    ABSENCE_CONFIRMED,
    ACTIVE,
    AMBIGUOUS,
    CREATING,
    HEALTH_SCHEMA,
    MAX_PROVIDER_DEADLINE_OBSERVATION_LAG_SECONDS,
    PROVIDER_DEADLINE_DRILL_MODE,
    PREPARED,
    TERMINAL,
    campaign_coordinates,
    GenerationConflict,
    LeaseConflict,
    LeaseError,
    LeaseStore,
    ReaperResult,
    install_systemd_user_timer,
    systemd_reaper_health,
    write_reaper_health,
    reap_once,
    runpod_authoritative_listing,
)
from fidelity.runpodapi import (  # noqa: E402
    DEFAULT_KEY_FILE, MIN_CREATE_SETUP_SECONDS, RunPod,
    RunPodCreateResponseError, RunPodError, _load_key, _strict_json_loads,
)

FAILED = []


def check(label, ok, detail=""):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)
        for line in str(detail).splitlines()[:10]:
            print("        %s" % line)


def raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return True
    return False

def write_resealed(path, document):
    unsealed = dict(document)
    unsealed.pop("record_sha256", None)
    raw = json.dumps(
        unsealed, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")
    unsealed["record_sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(
        json.dumps(
            unsealed, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8")
    path.chmod(0o600)

def server_time_fixture(epoch=1000.0):
    return {
        "schema": "fidelity-suite/runpod-server-time.v1",
        "endpoint_origin": "https://api.runpod.io",
        "date_header": "Thu, 01 Jan 1970 00:16:40 GMT",
        "server_epoch": epoch,
        "local_received_epoch": epoch,
        "local_minus_server_seconds": 0.0,
        "checked_at_epoch": epoch,
        "evidence_age_seconds": 0.0,
        "max_clock_delta_seconds": 30.0,
        "max_evidence_age_seconds": 30.0,
    }

def prepared_create_fixture(
        *, name, gpu, count, volume_gb, container_disk_gb,
        min_vcpu, min_ram_gb, image, terminate_after):
    identity = {
        "cloud_type": "SECURE", "is_spot": False, "offer": "on-demand",
        "gpu_type_id": gpu, "gpu_count": count, "volume_gb": volume_gb,
        "container_disk_gb": container_disk_gb, "min_vcpu": min_vcpu,
        "min_ram_gb": min_ram_gb, "name": name, "image_name": image,
        "terminate_after": terminate_after, "ports": "22/tcp",
        "volume_mount_path": "/workspace", "network_volume_id": None,
        "public_key_sha256": "0" * 64,
    }
    body = json.dumps({
        "query": "mutation { podFindAndDeployOnDemand }"
    }).encode("utf-8")
    return {
        "schema": "fidelity-suite/runpod-prepared-create.v1",
        "request_identity": identity,
        "graphql_body_sha256": hashlib.sha256(body).hexdigest(),
        "graphql_body_bytes": len(body),
        "graphql_body_base64": base64.b64encode(body).decode("ascii"),
    }


def begin(store, digit, provider="runpod", pre=()):
    prepared = store.begin_create(
        job_hash=digit * 64,
        provider=provider,
        request={"gpu": "A100", "count": 1},
        pre_create_resources=pre,
        create_deadline_epoch=store.clock() + 60,
        workload_deadline_epoch=store.clock() + 3600,
    )
    return store.record_post_intent(prepared)


def concurrent_begin(root, start, queue):
    store = LeaseStore(Path(root))
    start.wait()
    try:
        ref = begin(store, "a")
        queue.put(("created", ref.path.name))
    except Exception as exc:  # result is asserted by the parent
        queue.put((type(exc).__name__, str(exc)))


def lease_core_cases():
    print("== lease-v2 collision, generation, and response-lost reconciliation ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = LeaseStore(root / "collision")
        first = begin(store, "a")
        check("full hash and independent 96-bit attempt name the lease file",
              first.path.name.startswith("a" * 64 + ".")
              and len(first.attempt_id) == 24)

        prepared_store = LeaseStore(root / "prepared")
        prepared = prepared_store.begin_create(
            job_hash="7" * 64, provider="runpod",
            request={"gpu": "A100"}, pre_create_resources=[],
            create_deadline_epoch=prepared_store.clock() + 60,
            workload_deadline_epoch=prepared_store.clock() + 3600)
        check("begin_create proves no POST by returning PREPARED",
              prepared.state == PREPARED)
        cancelled = prepared_store.cancel_prepared(
            prepared, {"reason": "controller failed before POST intent"})
        check("PREPARED can close without absence or billing claims",
              cancelled.state == TERMINAL
              and prepared_store.read(cancelled)["provider_resource_ids"] == [])

        post_store = LeaseStore(root / "post-intent")
        post = post_store.begin_create(
            job_hash="6" * 64, provider="runpod",
            request={"gpu": "A100"}, pre_create_resources=[],
            create_deadline_epoch=post_store.clock() + 60,
            workload_deadline_epoch=post_store.clock() + 3600)
        post = post_store.record_post_intent(post)
        check("POST_INTENT is irrevocable and never pre-create-cancellable",
              post.state == CREATING
              and raises(LeaseError, lambda: post_store.cancel_prepared(
                  post, {"reason": "must remain ambiguous"})))

        safe_real = root / "safe-real"
        safe_real.mkdir(mode=0o700)
        symlink_root = root / "symlink-root"
        os.symlink(str(safe_real), str(symlink_root))
        check("lease state root symlink is refused",
              raises(LeaseError, lambda: begin(
                  LeaseStore(symlink_root), "5")))

        lock_root = root / "lock-symlink"
        lock_root.mkdir(mode=0o700)
        lock_victim = root / "lock-victim"
        lock_victim.write_text("unchanged\n", encoding="utf-8")
        lock_victim.chmod(0o600)
        os.symlink(
            str(lock_victim), str(lock_root / ("%s.lock" % ("4" * 64))))
        check("lease lock symlink is refused without touching target",
              raises(LeaseError, lambda: begin(
                  LeaseStore(lock_root), "4"))
              and lock_victim.read_text(encoding="utf-8") == "unchanged\n")

        lease_link_store = LeaseStore(root / "lease-symlink")
        lease_link = begin(lease_link_store, "3")
        lease_victim = root / "lease-victim"
        lease_victim.write_text("{}\n", encoding="utf-8")
        lease_victim.chmod(0o600)
        lease_link.path.unlink()
        os.symlink(str(lease_victim), str(lease_link.path))
        check("symlinked lease record is refused",
              raises(LeaseError, lambda: lease_link_store.read(lease_link)))

        duplicate_store = LeaseStore(root / "duplicate-json")
        duplicate = begin(duplicate_store, "2")
        duplicate.path.write_text(
            '{"schema":"x","schema":"y"}\n', encoding="utf-8")
        duplicate.path.chmod(0o600)
        check("duplicate lease JSON keys fail closed",
              raises(LeaseError, lambda: duplicate_store.read(duplicate)))
        unexpected_store = LeaseStore(root / "unexpected-key")
        unexpected = begin(unexpected_store, "b")
        unexpected_doc = unexpected_store.read(unexpected)
        unexpected_doc["ignored_evidence"] = {"claim": True}
        write_resealed(unexpected.path, unexpected_doc)
        check("self-sealed unexpected lease keys freeze the reaper",
              raises(LeaseError, lambda: reap_once(
                  unexpected_store, {"runpod": EmptyProvider()})))
        history_store = LeaseStore(root / "history-conflict")
        history_ref = begin(history_store, "c")
        history_doc = history_store.read(history_ref)
        history_doc["history"][-1]["generation"] = 0
        write_resealed(history_ref.path, history_doc)
        check("self-sealed conflicting generation history is refused",
              raises(LeaseError, lambda: history_store.read(history_ref)))
        check("an unresolved job cannot be overwritten by a second attempt",
              raises(LeaseConflict, lambda: begin(store, "a")))
        first_name = store.read(first)["create"]["exact_name"]
        active = store.record_create_success(
            first, {"id": "pod-one", "name": first_name})
        check("provider response binds exactly one new id",
              active.state == ACTIVE
              and store.read(active)["provider_resource_ids"] == ["pod-one"])
        check("stale generation cannot update a newer lease",
              raises(GenerationConflict,
                     lambda: store.record_create_success(
                         first, {"id": "pod-two", "name": first_name})))

        mismatch_store = LeaseStore(root / "mismatched-response")
        mismatch = begin(mismatch_store, "f")
        mismatch = mismatch_store.record_create_success(
            mismatch, {"id": "paid-pod", "name": "unexpected-name"})
        mismatch_doc = mismatch_store.read(mismatch)
        check("paid id is durable before post-create identity rejection",
              mismatch.state == ACTIVE
              and mismatch_doc["provider_resource_ids"] == ["paid-pod"]
              and (mismatch_doc["history"][-1]["evidence"]["response"]
                   ["name_matches_exact"]) is False)
        submitted_store = LeaseStore(root / "serialized-submit")
        submitted = begin(submitted_store, "e")
        submitted_name = submitted_store.read(
            submitted)["create"]["exact_name"]
        structured_error = None
        try:
            submitted_store.submit_create_and_record(
                submitted,
                lambda: (_ for _ in ()).throw(
                    RunPodCreateResponseError(
                        "fixture wrong-name response", "ack-pod", {
                            "id": "ack-pod",
                            "name": "provider-rewrote-name",
                            "cost_per_hr": "0.44",
                        })))
        except RunPodCreateResponseError as exc:
            structured_error = exc
        structured_ref = getattr(
            structured_error, "durable_lease_ref", None)
        check("structured create exception binds ID inside submission lock",
              structured_ref is not None
              and submitted_store.read(structured_ref)[
                  "provider_resource_ids"] == ["ack-pod"]
              and submitted_store.read(structured_ref)["state"] == ACTIVE)

        stale_submit_store = LeaseStore(root / "stale-submit")
        stale_submit = begin(stale_submit_store, "7")
        stale_submit_store.reconcile_response_lost(stale_submit, [])
        post_calls = []
        check("stale submit intent refuses before provider POST",
              raises(GenerationConflict, lambda:
                  stale_submit_store.submit_create_and_record(
                      stale_submit,
                      lambda: post_calls.append(True) or {
                          "id": "must-not-create",
                          "name": submitted_name,
                      }))
              and post_calls == [])


        # Two separate processes race the same full job hash.  The flock and
        # unresolved-attempt check must admit exactly one.
        race_root = root / "race"
        event = multiprocessing.Event()
        queue = multiprocessing.Queue()
        workers = [
            multiprocessing.Process(
                target=concurrent_begin, args=(str(race_root), event, queue))
            for _unused in range(2)
        ]
        for worker in workers:
            worker.start()
        event.set()
        outcomes = [queue.get(timeout=10) for _unused in workers]
        for worker in workers:
            worker.join(timeout=10)
        kinds = sorted(item[0] for item in outcomes)
        check("per-job flock admits one concurrent creator and refuses one",
              kinds == ["LeaseConflict", "created"], outcomes)

        clock = lambda: 1000.0
        zero_store = LeaseStore(root / "zero", clock=clock)
        zero = begin(zero_store, "b")
        zero_pending = zero_store.reconcile_response_lost(
            zero, [],
            response_error=("transport failed Authorization: Bearer "
                            "abcdefghijklmnopqrstuvwxyz"))
        check("lost response plus zero matches stays unresolved with redacted error",
              zero_pending.state == CREATING and zero_pending.path.exists()
              and "***REDACTED***" in
              zero_store.read(zero_pending)["history"][-1]["evidence"]
              ["response_error_redacted"])
        zero_closed = zero_store.reconcile_response_lost(
            zero_pending, [], create_window_closed=True)
        zero_again = zero_store.reconcile_response_lost(
            zero_closed, [], create_window_closed=True)
        check("zero matches after closed create window remains unresolved forever",
              zero_again.state == CREATING
              and zero_again.path.exists()
              and zero_store.read(zero_again)["history"][-1]["event"]
              == "LOST_CREATE_RESPONSE_RECONCILED_ZERO_WINDOW_CLOSED_UNRESOLVED")

        # A create the provider REFUSED by name is not a lost response. On
        # 2026-09-03T02:35Z a SUPPLY_CONSTRAINT refusal stranded a lease in
        # CREATING and closed the campaign's paid admission gate for good.
        refused_store = LeaseStore(root / "refused", clock=clock)
        refused = begin(refused_store, "d")
        refused_done = refused_store.reconcile_response_lost(
            refused, [], provider_rejection_codes=("SUPPLY_CONSTRAINT",),
            response_error="RunPod GraphQL create: SUPPLY_CONSTRAINT")
        refused_doc = refused_store.read(refused_done)
        # A refusal contradicted by a real pod must retain the liability:
        # the pod is bound for cleanup and the lease never closes terminally.
        contested_store = LeaseStore(root / "refused-contested", clock=clock)
        contested = begin(contested_store, "e")
        contested_name = contested_store.read(contested)["create"]["exact_name"]
        contested_done = contested_store.reconcile_response_lost(
            contested,
            [{"id": "surprise", "name": contested_name, "status": "RUNNING"}],
            provider_rejection_codes=("SUPPLY_CONSTRAINT",))
        check("named provider refusal closes only a lease with nothing attributable",
              refused_done.state == TERMINAL
              and refused_doc["history"][-1]["event"]
                  == "PROVIDER_REJECTED_CREATE_NO_RESOURCE"
              and refused_doc["history"][-1]["evidence"][
                  "provider_rejection_codes"] == ["SUPPLY_CONSTRAINT"]
              and refused_doc["terminal_proof"][
                  "provider_rejected_create"]["new_pod_ids"] == []
              and not refused_doc["provider_resource_ids"]
              and contested_done.state == ACTIVE
              and contested_store.read(contested_done)[
                  "provider_resource_ids"] == ["surprise"])

        # Only an enumerated code on a response carrying no id earns the
        # definitive classification; everything ambiguous stays fail-closed.
        classify = runpod_module._definitive_create_rejection_codes
        check("only a named, id-free provider refusal is definitive",
              classify({"errors": [{"extensions": {
                  "code": "SUPPLY_CONSTRAINT"}}]}) == ("SUPPLY_CONSTRAINT",)
              and classify({"errors": [{"extensions": {
                  "code": "SUPPLY_CONSTRAINT"}}],
                  "data": {"podFindAndDeployOnDemand": {"id": "pod-1"}}}) == ()
              and classify({"errors": [{"extensions": {
                  "code": "INTERNAL_SERVER_ERROR"}}]}) == ()
              and classify({"errors": [
                  {"extensions": {"code": "SUPPLY_CONSTRAINT"}},
                  {"message": "no extensions"}]}) == ()
              and classify({"errors": []}) == ()
              and classify({}) == ())

        one_store = LeaseStore(root / "one", clock=clock)
        one = begin(one_store, "c")
        one_name = one_store.read(one)["create"]["exact_name"]
        one_done = one_store.reconcile_response_lost(
            one, [{"id": "new-one", "name": one_name, "status": "RUNNING"}])
        check("lost response plus one exact new name binds that provider id",
              one_done.state == ACTIVE
              and one_store.read(one_done)["provider_resource_ids"] == ["new-one"])

        many_store = LeaseStore(root / "many", clock=clock)
        many = begin(many_store, "d")
        many_name = many_store.read(many)["create"]["exact_name"]
        many_done = many_store.reconcile_response_lost(
            many, [{"id": "new-a", "name": many_name},
                   {"id": "new-b", "name": many_name}])
        check("lost response plus multiple exact new names freezes ambiguous",
              many_done.state == AMBIGUOUS
              and many_done.path.exists()
              and many_store.read(many_done)["provider_resource_ids"]
              == ["new-a", "new-b"])

        wrong_store = LeaseStore(root / "wrong-family", clock=clock)
        wrong = begin(wrong_store, "8")
        wrong_name = wrong_store.read(wrong)["create"]["exact_name"]
        wrong_done = wrong_store.reconcile_response_lost(
            wrong, [{"id": "wrong-new", "name": "provider-rewrote-name"}],
            response_provider_id="response-id")
        wrong_evidence = wrong_store.read(wrong_done)["history"][-1]["evidence"]
        check("response loss never targets an unattributable wrong-name pod",
              wrong_done.state == AMBIGUOUS
              and wrong_store.read(wrong_done)["provider_resource_ids"]
              == ["response-id"]
              and wrong_evidence["wrong_name_new_pod_ids"] == ["wrong-new"]
              and wrong_evidence["unattributable_wrong_name_pod_ids"]
              == ["wrong-new"]
              and wrong_name == wrong_store.read(wrong_done)["create"]["exact_name"])
        acknowledged_store = LeaseStore(
            root / "acknowledged-wrong-name", clock=clock)
        acknowledged = begin(acknowledged_store, "5")
        acknowledged_done = acknowledged_store.reconcile_response_lost(
            acknowledged, [{
                "id": "acknowledged-id",
                "name": "provider-rewrote-name",
                "status": "RUNNING",
            }], response_provider_id="acknowledged-id")
        acknowledged_doc = acknowledged_store.read(acknowledged_done)
        acknowledged_evidence = (
            acknowledged_doc["terminal_proof"]["ambiguous_create"])
        acknowledged_destroying = acknowledged_store.request_destroy(
            acknowledged_done, {"reason": "selftest cleanup"})
        acknowledged_absent = acknowledged_store.confirm_exact_absence(
            acknowledged_destroying, [])
        check("acknowledged wrong-name response id remains deletable",
              acknowledged_done.state == AMBIGUOUS
              and acknowledged_doc["provider_resource_ids"]
                  == ["acknowledged-id"]
              and acknowledged_evidence[
                  "unattributable_wrong_name_pod_ids"] == []
              and acknowledged_absent.state == ABSENCE_CONFIRMED)
        wrong_only_store = LeaseStore(root / "wrong-only-family", clock=clock)
        wrong_only = begin(wrong_only_store, "6")
        wrong_only_done = wrong_only_store.reconcile_response_lost(
            wrong_only,
            [{"id": "not-ours", "name": "unrelated-concurrent-pod"}])
        wrong_only_doc = wrong_only_store.read(wrong_only_done)
        check("wrong-name-only delta freezes without a cleanup target",
              wrong_only_done.state == AMBIGUOUS
              and wrong_only_doc["provider_resource_ids"] == []
              and wrong_only_doc["terminal_proof"]["ambiguous_create"]
              ["unattributable_wrong_name_pod_ids"] == ["not-ours"])

        volume_store = LeaseStore(root / "volume-family", clock=clock)
        volume = begin(volume_store, "9")
        volume_done = volume_store.reconcile_response_lost(
            volume, [], network_volumes=[
                {"id": "surprise-volume", "name": "unexpected"}])
        check("surprise network-volume delta remains ambiguous blocker",
              volume_done.state == AMBIGUOUS
              and raises(
                  LeaseError,
                  lambda: volume_store.confirm_exact_absence(volume_done, []))
              and volume_store.read(volume_done)["terminal_proof"]
              ["ambiguous_create"]["new_network_volume_ids"]
              == ["surprise-volume"])
        post_store = LeaseStore(root / "post-create-family", clock=clock)
        post_ref = begin(post_store, "7")
        post_name = post_store.read(post_ref)["create"]["exact_name"]
        post_ref = post_store.record_create_success(
            post_ref, {"id": "intended-pod", "name": post_name})
        post_ref = post_store.bind_post_create_inventory(
            post_ref,
            [{"id": "intended-pod", "name": post_name},
             {"id": "extra-pod", "name": "provider-extra"}],
            network_volumes=[{"id": "extra-volume", "name": "surprise"}])
        post_doc = post_store.read(post_ref)
        check("post-create anomalies target only the attributable pod",
              post_ref.state == AMBIGUOUS
              and post_doc["provider_resource_ids"] == ["intended-pod"]
              and post_doc["terminal_proof"]["ambiguous_create"]
              ["unattributable_wrong_name_pod_ids"] == ["extra-pod"]
              and post_doc["terminal_proof"]["ambiguous_create"]
              ["new_network_volume_ids"] == ["extra-volume"])


def store_or(document, *keys):
    value = document
    for key in keys:
        value = (value or {}).get(key)
    return value


class OutageProvider:
    def list_instances(self):
        raise RunPodError("provider unavailable")


class EmptyProvider:
    def list_instances(self):
        return []

    def reconcile_billing(self, lease):
        return {"reconciled": True, "provider": lease["create"]["provider"],
                "evidence": "authoritative empty-create billing query"}

class StatefulProvider:
    def __init__(
            self, instances, account_id="acct-test", *,
            rest_instances=None, network_volumes=None,
            inventory_complete=True):
        self.instances = list(instances)
        self.rest_instances = list(
            instances if rest_instances is None else rest_instances)
        self.network_volumes = list(network_volumes or [])
        self.destroyed = []
        self.account_id = account_id
        self.billing_reads = 0
        self.inventory_complete = inventory_complete

    def status(self):
        return {"id": self.account_id, "clientBalance": "100.00"}

    def list_instances(self):
        return list(self.instances)

    def chargeable_inventory(self):
        complete = self.inventory_complete
        return {
            "schema": "fidelity-suite/runpod-chargeable-inventory.v1",
            "provider": "runpod",
            "observed_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "complete": complete,
            "unknown_families": [] if complete else ["pods"],
            "families": {
                "pods": {
                    "complete": complete,
                    "source": "test REST pods",
                    "resources": list(self.rest_instances) if complete else [],
                },
                "network_volumes": {
                    "complete": True,
                    "source": "test REST network volumes",
                    "resources": list(self.network_volumes),
                },
            },
        }

    def destroy(self, provider_id):
        self.destroyed.append(str(provider_id))
        self.instances = [
            item for item in self.instances
            if str(item["id"]) != str(provider_id)
        ]
        self.rest_instances = [
            item for item in self.rest_instances
            if str(item["id"]) != str(provider_id)
        ]

    def reconcile_billing(self, lease):
        self.billing_reads += 1
        return {
            "reconciled": True,
            "provider": lease["create"]["provider"],
            "provider_resource_ids": lease["provider_resource_ids"],
            "total_amount": "3.00",
            "evidence": {
                "schema": "fidelity-suite/runpod-billing-retrieval.v1",
                "retrieval_id": "%024x" % self.billing_reads,
                "retrieved_at_utc": "1970-01-01T01:00:00Z",
            },
        }

class PartialThenIncreasesProvider(StatefulProvider):
    def reconcile_billing(self, lease):
        self.billing_reads += 1
        return {
            "reconciled": True,
            "provider": lease["create"]["provider"],
            "provider_resource_ids": lease["provider_resource_ids"],
            "total_amount": str(self.billing_reads),
            "evidence": {
                "schema": "fidelity-suite/runpod-billing-retrieval.v1",
                "retrieval_id": "%024x" % self.billing_reads,
                "retrieved_at_utc": "1970-01-01T01:00:00Z",
            },
        }


def copy_reaper_source_tree(destination):
    destination = Path(destination)
    destination.mkdir(mode=0o700)
    fidelity = destination / "fidelity"
    fidelity.mkdir(mode=0o700)
    shutil.copy2(ROOT / "bin" / "reap_cloud_leases.py",
                 destination / "reap_cloud_leases.py")
    for name in (
            "__init__.py", "cloudlease.py", "campaign.py", "common.py",
            "runpodapi.py", "jlapi.py", "sshbase.py"):
        shutil.copy2(ROOT / "bin" / "fidelity" / name, fidelity / name)
    for path in [destination / "reap_cloud_leases.py"] + list(
            fidelity.iterdir()):
        path.chmod(0o644)
    return destination / "reap_cloud_leases.py"


def reaper_cases():
    print("\n== exact absence, EXITED-is-live, and provider isolation ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        prepared_reaper_store = LeaseStore(root / "prepared-reaper")
        prepared_reaper = prepared_reaper_store.begin_create(
            job_hash="5" * 64, provider="runpod",
            request={"gpu": "A100"}, pre_create_resources=[],
            create_deadline_epoch=prepared_reaper_store.clock() + 60,
            workload_deadline_epoch=prepared_reaper_store.clock() + 3600,
            controller_pid=2 ** 30)
        prepared_result = reap_once(
            prepared_reaper_store, {}, now=1001.0)
        check("reaper cancels PREPARED without consulting provider inventory",
              prepared_reaper_store.read(prepared_reaper)["state"] == TERMINAL
              and prepared_result.ok)
        live_store = LeaseStore(
            root / "prepared-live", clock=lambda: 1000.0)
        live_prepared = live_store.begin_create(
            job_hash="6" * 64, provider="runpod",
            request={"gpu": "A100"}, pre_create_resources=[],
            create_deadline_epoch=1060, workload_deadline_epoch=4600,
            controller_pid=os.getpid())
        live_result = reap_once(live_store, {}, now=1001.0)
        check("live same-UID PREPARED controller is never cancelled",
              live_store.read(live_prepared)["state"] == PREPARED
              and live_result.ok
              and any(action["action"]
                      == "prepared-controller-active-deferred"
                      for action in live_result.actions))
        expired_live_result = reap_once(live_store, {}, now=2000.0)
        check("expired PREPARED deadline cannot race a live controller",
              live_store.read(live_prepared)["state"] == PREPARED
              and expired_live_result.ok
              and any(action["action"]
                      == "prepared-controller-active-deferred"
                      for action in expired_live_result.actions))
        unknown_store = LeaseStore(
            root / "prepared-owner-unknown", clock=lambda: 1000.0)
        unknown_prepared = unknown_store.begin_create(
            job_hash="3" * 64, provider="runpod",
            request={"gpu": "A100"}, pre_create_resources=[],
            create_deadline_epoch=1001, workload_deadline_epoch=4600,
            controller_pid=None)
        unknown_result = reap_once(unknown_store, {}, now=2000.0)
        check("unknown PREPARED owner state defers indefinitely without spend",
              unknown_store.read(unknown_prepared)["state"] == PREPARED
              and unknown_result.ok
              and any(action["action"]
                      == "prepared-controller-active-deferred"
                      for action in unknown_result.actions))
        orphan_ledger_path = root / "orphan-campaign.json"
        CampaignLedger.create(
            str(orphan_ledger_path), "100", "10", "2",
            max_concurrent_attempts=2, provider="runpod",
            provider_account_id="acct-test")
        orphan_store = LeaseStore(
            root / "orphan-prepared", clock=lambda: 1000.0)
        orphan_job = "4" * 64
        orphan_attempt = "d" * 24
        orphan_name = "fidcloud-%s-a%s" % (
            orphan_job, orphan_attempt)
        orphan_request = {
            "attempt_key": orphan_attempt,
            "campaign_attempt_key": attempt_key(
                orphan_job, orphan_attempt),
            "campaign_ledger": orphan_ledger_path.name,
            "provider": "runpod",
            "provider_account_id": "acct-test",
            "gpu_type": "A100", "normalized_gpu": "A100",
            "num_gpus": 1, "secure_cloud": True,
            "storage_gb": 100, "remote_root": "/workspace/f",
            "engine_root": "/workspace/e", "container_disk_gb": 20,
            "image": "image", "min_vcpu_count": 4,
            "min_memory_gb": 16, "workload_contract": {},
            "offer": "on-demand", "network_volume": None,
            "terminate_after": "2030-01-02T03:04:05Z", "quote": {},
            "pre_create_safety": server_time_fixture(),
            "execution_contract_sha256": "d" * 64,
            "grounding_bundle": {
                "schema": "fidelity-suite/grounding-bundle.v1",
                "archive_sha256": "e" * 64,
                "archive_bytes": 1,
                "manifest_sha256": "f" * 64,
            },
            "prepared_create": prepared_create_fixture(
                name=orphan_name, gpu="A100", count=1,
                volume_gb=100, container_disk_gb=20,
                min_vcpu=4, min_ram_gb=16, image="image",
                terminate_after="2030-01-02T03:04:05Z"),
        }
        orphan_ref = orphan_store.begin_create(
            job_hash=orphan_job, provider="runpod",
            request=orphan_request, pre_create_resources=[],
            attempt_id=orphan_attempt, controller_pid=2 ** 30,
            create_deadline_epoch=1060,
            workload_deadline_epoch=4600)
        orphan_result = reap_once(orphan_store, {}, now=1001.0)
        orphan_document = orphan_store.read(orphan_ref)
        check("crash after PREPARED but before campaign reserve closes safely",
              orphan_result.ok
              and orphan_document["state"] == TERMINAL
              and orphan_document["create"]["controller_pid"] == 2 ** 30
              and CampaignLedger(
                  str(orphan_ledger_path), "runpod",
                  "acct-test").snapshot()["attempts"] == {})
        partial_store = LeaseStore(root / "partial-billing", clock=lambda: 1000.0)
        partial_ref = begin(partial_store, "0")
        partial_name = partial_store.read(partial_ref)["create"]["exact_name"]
        partial_ref = partial_store.record_create_success(
            partial_ref, {"id": "partial-pod", "name": partial_name})
        partial_ref = partial_store.confirm_exact_absence(partial_ref, [])
        partial_result = reap_once(
            partial_store,
            {"runpod": PartialThenIncreasesProvider([], "acct-test")},
            now=1301.0)
        # The pod is proven absent, so unstable billing is a pending action
        # the next sweep retries, never a failure that deletes reaper health.
        check("partial billing that increases across retrievals stays pending",
              partial_store.read(partial_ref)["state"] == ABSENCE_CONFIRMED
              and partial_store.read(partial_ref)["billing_reconciliation"] is None
              and partial_result.ok
              and any(action["action"] == "billing-pending"
                      and "changed between stabilization retrievals"
                      in action["reason"]
                      for action in partial_result.actions))
        store = LeaseStore(root / "exited", clock=lambda: 1000.0)
        ref = begin(store, "e")
        exact_name = store.read(ref)["create"]["exact_name"]
        ref = store.record_create_success(
            ref, {"id": "pod-exited", "name": exact_name})
        still_live = store.confirm_exact_absence(
            ref, [{"id": "pod-exited", "name": "anything", "status": "EXITED"}])
        check("EXITED remains live while its exact id is still listed",
              still_live.state == ACTIVE
              and still_live.path.exists()
              and store.read(still_live)["history"][-1]["event"]
              == "EXACT_IDS_STILL_LISTED")

        exited_provider = StatefulProvider([
            {"id": "pod-exited", "name": exact_name, "status": "EXITED"}])
        exited_result = reap_once(
            store, {"runpod": exited_provider}, now=1001.0)
        check("listed EXITED is deleted immediately; settlement awaits billing stabilization",
              exited_provider.destroyed == ["pod-exited"]
              and store.read(still_live)["state"] == ABSENCE_CONFIRMED
              and any(action["action"] == "billing-stabilization-waiting"
                      for action in exited_result.actions)
              and exited_result.ok, exited_result.to_dict())
        disagreement_store = LeaseStore(
            root / "graphql-rest-disagreement", clock=lambda: 1000.0)
        disagreement_ref = begin(disagreement_store, "7")
        disagreement_name = disagreement_store.read(
            disagreement_ref)["create"]["exact_name"]
        disagreement_ref = disagreement_store.record_create_success(
            disagreement_ref,
            {"id": "rest-only-pod", "name": disagreement_name})
        disagreement_provider = StatefulProvider(
            [], rest_instances=[{
                "id": "rest-only-pod", "name": disagreement_name,
                "status": "RUNNING"}])
        active_disagreement = reap_once(
            disagreement_store, {"runpod": disagreement_provider}, now=1001.0)
        check("REST-live GraphQL omission cannot confirm ACTIVE absence",
              disagreement_store.read(disagreement_ref)["state"] == ACTIVE
              and disagreement_provider.destroyed == []
              and active_disagreement.ok)
        disagreement_ref = disagreement_store.request_destroy(
            disagreement_ref, {
                "reason": "deterministic disagreement cleanup",
                "provider_ids": ["rest-only-pod"],
            })
        destroying_disagreement = reap_once(
            disagreement_store, {"runpod": disagreement_provider}, now=1001.0)
        check("DESTROYING uses REST-only exact id and later confirms both views",
              disagreement_provider.destroyed == ["rest-only-pod"]
              and disagreement_store.read(disagreement_ref)["state"]
              == ABSENCE_CONFIRMED
              and destroying_disagreement.ok)
        conflict_provider = StatefulProvider(
            [{"id": "conflict-pod", "name": "graphql-name",
              "status": "RUNNING"}],
            rest_instances=[{
                "id": "conflict-pod", "name": "rest-name",
                "status": "RUNNING"}])
        check("direct cleanup refuses GraphQL/REST identity disagreement",
              raises(
                  LeaseError,
                  lambda: runpod_authoritative_listing(
                      conflict_provider, conflict_provider.list_instances(),
                      "acct-test",
                      inventory=conflict_provider.chargeable_inventory())))

        acknowledged_store = LeaseStore(
            root / "acknowledged-rest-duplicate", clock=lambda: 1000.0)
        acknowledged_ref = begin(acknowledged_store, "a")
        acknowledged_name = acknowledged_store.read(
            acknowledged_ref)["create"]["exact_name"]
        acknowledged_ref = acknowledged_store.record_create_success(
            acknowledged_ref,
            {"id": "acknowledged-pod", "name": acknowledged_name})
        acknowledged_provider = StatefulProvider(
            [{"id": "acknowledged-pod", "name": acknowledged_name,
              "status": "RUNNING"}],
            rest_instances=[
                {"id": "acknowledged-pod", "name": acknowledged_name,
                 "status": "RUNNING"},
                {"id": "rest-only-duplicate", "name": acknowledged_name,
                 "status": "RUNNING"},
            ])
        acknowledged_union, unused_proof = runpod_authoritative_listing(
            acknowledged_provider, acknowledged_provider.list_instances(),
            "acct-test",
            inventory=acknowledged_provider.chargeable_inventory())
        acknowledged_ref = acknowledged_store.bind_post_create_inventory(
            acknowledged_ref, acknowledged_union)
        check("acknowledged create tracks REST-only exact-name duplicate",
              acknowledged_ref.state == AMBIGUOUS
              and acknowledged_store.read(acknowledged_ref)[
                  "provider_resource_ids"]
              == ["acknowledged-pod", "rest-only-duplicate"])

        response_lost_store = LeaseStore(
            root / "response-lost-rest-only", clock=lambda: 1000.0)
        response_lost_ref = begin(response_lost_store, "b")
        response_lost_name = response_lost_store.read(
            response_lost_ref)["create"]["exact_name"]
        response_lost_provider = StatefulProvider(
            [], rest_instances=[{
                "id": "rest-only-response-lost",
                "name": response_lost_name, "status": "RUNNING"}])
        response_lost_result = reap_once(
            response_lost_store, {"runpod": response_lost_provider},
            now=1001.0)
        check("response-lost create tracks REST-only exact-name candidate",
              response_lost_store.read(response_lost_ref)["state"] == ACTIVE
              and response_lost_store.read(response_lost_ref)[
                  "provider_resource_ids"] == ["rest-only-response-lost"]
              and response_lost_result.ok)

        wrong_rest_store = LeaseStore(
            root / "response-lost-rest-wrong-name", clock=lambda: 1000.0)
        wrong_rest_ref = begin(wrong_rest_store, "c")
        wrong_rest_provider = StatefulProvider(
            [], rest_instances=[{
                "id": "rest-only-wrong-name",
                "name": "unrelated-name", "status": "RUNNING"}])
        wrong_rest_result = reap_once(
            wrong_rest_store, {"runpod": wrong_rest_provider}, now=1001.0)
        wrong_rest_document = wrong_rest_store.read(wrong_rest_ref)
        # Nothing attributable means nothing this reaper may delete; the
        # lease is reported for an operator and the sweep stays healthy.
        check("REST-only wrong-name delta remains unbound and needs an operator",
              wrong_rest_document["state"] == AMBIGUOUS
              and wrong_rest_document["provider_resource_ids"] == []
              and wrong_rest_document["terminal_proof"]["ambiguous_create"]
                  ["unattributable_wrong_name_pod_ids"]
                  == ["rest-only-wrong-name"]
              and wrong_rest_result.ok
              and any(action["action"] == "ambiguous-needs-operator"
                      for action in wrong_rest_result.actions))
        volume_rest_store = LeaseStore(
            root / "response-lost-rest-volume", clock=lambda: 1000.0)
        volume_rest_ref = begin(volume_rest_store, "d")
        volume_rest_provider = StatefulProvider(
            [], rest_instances=[], network_volumes=[{
                "id": "rest-only-volume", "name": "persistent-volume"}])
        volume_rest_result = reap_once(
            volume_rest_store, {"runpod": volume_rest_provider}, now=1001.0)
        volume_rest_document = volume_rest_store.read(volume_rest_ref)
        check("response-loss classification includes exact REST volume family",
              volume_rest_document["state"] == AMBIGUOUS
              and volume_rest_document["provider_resource_ids"] == []
              and volume_rest_document["terminal_proof"]["ambiguous_create"]
                  ["new_network_volume_ids"] == ["rest-only-volume"]
              and volume_rest_result.ok
              and any(action["action"] == "ambiguous-needs-operator"
                      for action in volume_rest_result.actions))

        legacy_store = LeaseStore(
            root / "legacy-absence-disagreement", clock=lambda: 1000.0)
        legacy_ref = begin(legacy_store, "8")
        legacy_name = legacy_store.read(legacy_ref)["create"]["exact_name"]
        legacy_ref = legacy_store.record_create_success(
            legacy_ref, {"id": "legacy-rest-pod", "name": legacy_name})
        legacy_ref = legacy_store.confirm_exact_absence(legacy_ref, [])
        legacy_provider = StatefulProvider(
            [], rest_instances=[{
                "id": "legacy-rest-pod", "name": legacy_name,
                "status": "RUNNING"}])
        legacy_result = reap_once(
            legacy_store, {"runpod": legacy_provider}, now=1301.0)
        legacy_document = legacy_store.read(legacy_ref)
        check("legacy false absence reopens, deletes, and terminally reconciles",
              legacy_provider.destroyed == ["legacy-rest-pod"]
              and legacy_document["state"] == TERMINAL
              and any(event["event"] == "ABSENCE_PROOF_REVOKED"
                      for event in legacy_document["history"])
              and legacy_result.ok and not legacy_result.unresolved,
              legacy_result.to_dict())
        account_store = LeaseStore(
            root / "account-binding", clock=lambda: 1000.0)
        account_ref = account_store.begin_create(
            job_hash="9" * 64, provider="runpod",
            request={
                "attempt_key": "a" * 24,
                "campaign_attempt_key": "attempt-account",
                "campaign_ledger": "campaign.json",
                "provider": "runpod",
                "provider_account_id": "acct-test",
                "gpu_type": "A100", "normalized_gpu": "A100",
                "num_gpus": 1, "secure_cloud": True,
                "storage_gb": 100, "remote_root": "/workspace/f",
                "engine_root": "/workspace/e", "container_disk_gb": 20,
                "image": "image", "min_vcpu_count": 4,
                "min_memory_gb": 16, "workload_contract": {},
                "offer": "on-demand", "network_volume": None,
                "terminate_after": "2030-01-02T03:04:05Z", "quote": {},
                "pre_create_safety": server_time_fixture(),
                "execution_contract_sha256": "d" * 64,
                "grounding_bundle": {
                    "schema": "fidelity-suite/grounding-bundle.v1",
                    "archive_sha256": "e" * 64,
                    "archive_bytes": 1,
                    "manifest_sha256": "f" * 64,
                },
                "prepared_create": prepared_create_fixture(
                    name="fidcloud-%s-a%s" % ("9" * 64, "a" * 24),
                    gpu="A100", count=1, volume_gb=100,
                    container_disk_gb=20, min_vcpu=4, min_ram_gb=16,
                    image="image",
                    terminate_after="2030-01-02T03:04:05Z"),
            },
            pre_create_resources=[],
            attempt_id="a" * 24,
            create_deadline_epoch=1060, workload_deadline_epoch=4600)
        account_ref = account_store.record_post_intent(account_ref)
        coordinates = campaign_coordinates(
            account_store.read(account_ref), account_store.root)
        check("campaign locator resolves a safe leaf beside the lease root",
              coordinates == (
                  account_store.root.parent / "campaign.json",
                  "attempt-account"))
        account_ref = account_store.record_create_success(
            account_ref, {"id": "pod-account", "name":
                          account_store.read(account_ref)["create"]["exact_name"]})
        wrong_account = StatefulProvider(
            [{"id": "pod-account", "name": "anything"}],
            account_id="other-account")
        account_result = reap_once(
            account_store, {"runpod": wrong_account}, now=1001.0)
        check("provider account mismatch freezes deletion and remains unresolved",
              wrong_account.destroyed == []
              and account_store.read(account_ref)["state"] == ACTIVE
              and not account_result.ok
              and account_ref.path.name in account_result.unresolved)
        retained_provider = StatefulProvider(
            [{"id": "pod-account", "name": "anything", "status": "RUNNING"}],
            account_id="acct-test")
        retained_result = reap_once(
            account_store, {"runpod": retained_provider}, now=5000.0)
        retained_document = account_store.read(account_ref)
        check("running pod survives workload deadline through retrieval reserve",
              retained_provider.destroyed == []
              and retained_document["state"] == ACTIVE
              and retained_document["create"]["reap_deadline_utc"]
              == retained_document["create"]["request"]["terminate_after"]
              and retained_document["create"]["reap_deadline_epoch"]
              > retained_document["create"]["workload_deadline_epoch"]
              and retained_result.ok)

        ambiguous_store = LeaseStore(
            root / "ambiguous-cleanup", clock=lambda: 1000.0)
        ambiguous = begin(ambiguous_store, "2")
        ambiguous_name = ambiguous_store.read(ambiguous)["create"]["exact_name"]
        ambiguous = ambiguous_store.reconcile_response_lost(
            ambiguous,
            [{"id": "candidate-b", "name": ambiguous_name, "status": "RUNNING"},
             {"id": "candidate-a", "name": ambiguous_name, "status": "EXITED"}])
        ambiguous_provider = StatefulProvider([
            {"id": "candidate-a", "name": ambiguous_name, "status": "EXITED"},
            {"id": "candidate-b", "name": ambiguous_name, "status": "RUNNING"}])
        ambiguous_result = reap_once(
            ambiguous_store, {"runpod": ambiguous_provider}, now=1001.0)
        ambiguous_doc = ambiguous_store.read(ambiguous)
        check("ambiguous create deletes every attributable candidate; settlement waits",
              ambiguous_provider.destroyed == ["candidate-a", "candidate-b"]
              and ambiguous_doc["state"] == ABSENCE_CONFIRMED
              and ambiguous_doc["terminal_proof"]["ambiguous_create"]
              ["new_exact_name_ids"] == ["candidate-a", "candidate-b"]
              and any(action["action"] == "billing-stabilization-waiting"
                      for action in ambiguous_result.actions)
              and ambiguous_result.ok, ambiguous_result.to_dict())
        check("empty provider ids are rejected as incomplete inventory",
              raises(LeaseError, lambda: begin(
                  LeaseStore(root / "empty-id", clock=lambda: 1000.0),
                  "3", pre=[{"id": "", "name": "invalid"}])))

        late_store = LeaseStore(
            root / "late-create", clock=lambda: 1000.0)
        late = begin(late_store, "4")
        late = late_store.reconcile_response_lost(
            late, [], create_window_closed=True)
        late = late_store.reconcile_response_lost(
            late, [], create_window_closed=True)
        late_name = late_store.read(late)["create"]["exact_name"]
        late_provider = StatefulProvider([
            {"id": "late-pod", "name": late_name, "status": "RUNNING"}])
        late_result = reap_once(
            late_store, {"runpod": late_provider}, now=5000.0)
        check("late appearance after repeated zero listings binds then destroys",
              late_provider.destroyed == ["late-pod"]
              and late_store.read(late)["state"] == TERMINAL
              and not late_result.unresolved,
              late_result.to_dict())

        # A lost create response with nothing attributable across complete
        # listings expires TERMINAL once the window has been closed for
        # LOST_CREATE_EXPIRY_SECONDS; before that it stays CREATING.  On
        # 2026-09-03 one such lease parked forever and closed paid admission
        # for the whole campaign while its liability was zero throughout.
        from fidelity.cloudlease import LOST_CREATE_EXPIRY_SECONDS
        expiry_store = LeaseStore(root / "expiry", clock=lambda: 1000.0)
        expiring = begin(expiry_store, "5")
        # Sweep 1: past the bound but the FIRST closed-window listing; must
        # not expire on a single look.  Sweep 2: a second independent complete
        # listing shows nothing attributable; expires with the release hook.
        first_look = reap_once(
            expiry_store, {"runpod": StatefulProvider([])},
            now=1060.0 + LOST_CREATE_EXPIRY_SECONDS)
        first_state = expiry_store.read(expiring)["state"]
        late_result = reap_once(
            expiry_store, {"runpod": StatefulProvider([])},
            now=1060.0 + LOST_CREATE_EXPIRY_SECONDS + 300)
        expired_document = expiry_store.read(expiring)
        check("lost create expires TERMINAL only on a second closed-window "
              "listing past the expiry bound",
              first_look.ok and first_state == CREATING
              and late_result.ok
              and expired_document["state"] == TERMINAL
              and expired_document["provider_resource_ids"] == []
              and expired_document["terminal_proof"]["lost_create_expired"]
                  ["expired_after_seconds"]
                  == LOST_CREATE_EXPIRY_SECONDS + 300
              and any(action["action"] == "lost-create-expired"
                      and action["campaign_release"] is None
                      for action in late_result.actions)
              and expiring.path.name not in late_result.unresolved,
              late_result.to_dict())
        # A release hook that raises leaves the lease CREATING for the next
        # sweep, so a TERMINAL lease never strands an unreleased reservation.
        hook_store = LeaseStore(root / "expiry-hook", clock=lambda: 1000.0)
        hooked = begin(hook_store, "6")
        hooked = hook_store.reconcile_response_lost(
            hooked, [], create_window_closed=True)

        def refusing_hook(unused_document, unused_evidence):
            raise LeaseError("ledger unavailable")

        check("expiry is withheld when the campaign release raises",
              raises(LeaseError, lambda: hook_store.reconcile_response_lost(
                  hooked, [], create_window_closed=True,
                  seconds_since_window_closed=LOST_CREATE_EXPIRY_SECONDS,
                  before_expire=refusing_hook))
              and hook_store.read(hooked)["state"] == CREATING)

        mixed = LeaseStore(root / "mixed", clock=lambda: 1000.0)
        outage = begin(mixed, "f", provider="outage")
        healthy = begin(mixed, "1", provider="healthy")
        result = reap_once(
            mixed, {"outage": OutageProvider(), "healthy": EmptyProvider()},
            now=2000.0)
        check("one provider outage is reported and preserves its lease",
              not result.ok and mixed.read(outage)["state"] == CREATING
              and outage.path.name in result.unresolved, result.to_dict())
        # The healthy provider's lease is still processed: its first
        # closed-window listing is recorded and it stays CREATING.
        check("provider outage does not block another provider's continued scan",
              mixed.read(healthy)["state"] == CREATING
              and mixed.read(healthy)["history"][-1]["event"]
                  == "LOST_CREATE_RESPONSE_RECONCILED_ZERO_WINDOW_CLOSED_UNRESOLVED"
              and healthy.path.name in result.unresolved,
              result.to_dict())
        secure_parent = Path(tempfile.mkdtemp(
            prefix="fidelity-reaper-test-", dir=str(Path.home())))
        secure_parent.chmod(0o700)
        health_dir = secure_parent / "health"
        lease_health_dir = health_dir / "leases-v2"
        source_parent = secure_parent / "sources"
        source_parent.mkdir(mode=0o700)
        unit_dir = root / "units"
        systemctl = root / "fake-systemctl"
        systemctl_log = root / "fake-systemctl.log"
        systemctl.write_text(
            "#!/bin/sh\n"
            "printf '%%s\\n' \"$*\" >> '%s'\n"
            "if [ \"$2\" = is-enabled ]; then echo enabled; fi\n"
            "if [ \"$2\" = is-active ]; then echo active; fi\n"
            "if [ \"$2\" = show ]; then "
            "printf 'Result=success\\nExecMainStatus=0\\n'; fi\n"
            "exit 0\n" % systemctl_log, encoding="utf-8")
        systemctl.chmod(0o700)
        linger_yes = root / "fake-loginctl-yes"
        linger_yes.write_text(
            "#!/bin/sh\nprintf 'yes\\n'\n", encoding="utf-8")
        linger_yes.chmod(0o700)
        linger_no = root / "fake-loginctl-no"
        linger_no.write_text(
            "#!/bin/sh\nprintf 'no\\n'\n", encoding="utf-8")
        linger_no.chmod(0o700)
        source_entry = copy_reaper_source_tree(source_parent / "trusted")
        source_command = [
            sys.executable, str(source_entry),
            "--provider", "runpod", "--sweep",
            "--lease-dir", str(lease_health_dir),
            "--reaper-state-dir", str(health_dir),
            "--runpod-key-file", str(root / "key"),
        ]
        install_systemd_user_timer(
            source_command, lease_dir=lease_health_dir,
            provider="runpod", provider_account_id="acct-test",
            state_dir=health_dir, unit_dir=unit_dir,
            systemctl=str(systemctl), loginctl=str(linger_yes))
        healthy_result = ReaperResult(
            ok=True, actions=tuple(), failures=tuple(), unresolved=tuple())
        health_now = time.time()
        check("mutable process cannot call health writer directly",
              raises(LeaseError, lambda: write_reaper_health(
                  health_dir, healthy_result, lease_dir=lease_health_dir,
                  provider="runpod", provider_account_id="acct-test",
                  now=health_now))
              and not (health_dir / "reaper-health-runpod.json").exists())
        real_runtime_verifier = (
            cloudlease_module.verify_reaper_control_account)
        cloudlease_module.verify_reaper_control_account = (
            lambda state_dir, **kwargs:
            cloudlease_module._verified_control(
                state_dir, lease_dir=kwargs["lease_dir"],
                provider=kwargs["provider"],
                provider_account_id=kwargs["provider_account_id"]))
        try:
            stamp_path = write_reaper_health(
                health_dir, healthy_result, lease_dir=lease_health_dir,
                provider="runpod", provider_account_id="acct-test",
                now=health_now)
        finally:
            cloudlease_module.verify_reaper_control_account = (
                real_runtime_verifier)
        stamp_before_checkout_entry = stamp_path.read_bytes()
        direct_checkout = subprocess.run(
            source_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30)
        check("mutable checkout entry cannot author installed reaper health",
              direct_checkout.returncode == 90
              and "installed snapshot entrypoint" in direct_checkout.stderr
              and stamp_path.read_bytes() == stamp_before_checkout_entry)
        health_args = {
            "state_dir": health_dir, "lease_dir": lease_health_dir,
            "provider": "runpod", "provider_account_id": "acct-test",
            "systemctl": str(systemctl), "now": health_now + 1,
            "max_age_seconds": 60,
        }
        health = systemd_reaper_health(
            loginctl=str(linger_yes), **health_args)
        failed_systemctl = root / "failed-systemctl"
        failed_systemctl.write_text(
            "#!/bin/sh\n"
            "if [ \"$2\" = is-enabled ]; then echo enabled; fi\n"
            "if [ \"$2\" = is-active ]; then echo active; fi\n"
            "if [ \"$2\" = show ]; then "
            "printf 'Result=exit-code\\nExecMainStatus=2\\n'; fi\n"
            "exit 0\n", encoding="utf-8")
        failed_systemctl.chmod(0o700)
        failed_health_args = dict(
            health_args, systemctl=str(failed_systemctl))
        latest_service_failed = systemd_reaper_health(
            loginctl=str(linger_yes), **failed_health_args)
        stamp_raw = stamp_path.read_text(encoding="utf-8")
        template_service = unit_dir / "fidelity-cloud-reaper@.service"
        template_service_text = template_service.read_text(encoding="utf-8")
        dropin = (unit_dir / "fidelity-cloud-reaper@runpod.service.d"
                  / "override.conf")
        dropin_text = dropin.read_text(encoding="utf-8")
        control_manifest = json.loads(
            (health_dir / "reaper-control-runpod.json").read_text(
                encoding="utf-8"))
        source_names = {
            Path(path).name for path in control_manifest["source_paths"]}
        runtime_names = {
            Path(path).name for path in control_manifest["runtime_paths"]}
        runtime_entry = Path(control_manifest["command_argv"][3])
        installed_entry_bytes = runtime_entry.read_bytes()
        public_runtime_names = {
            row["path"] for row in
            health["stamp"]["control"]["runtime_files"]}
        expected_public_names = {
            "bin/reap_cloud_leases.py",
            "bin/fidelity/__init__.py",
            "bin/fidelity/cloudlease.py",
            "bin/fidelity/campaign.py",
            "bin/fidelity/common.py",
            "bin/fidelity/jlapi.py",
            "bin/fidelity/runpodapi.py",
            "bin/fidelity/sshbase.py",
        }
        # Control seals the runtime snapshot the timer executes.  The
        # checkout it was copied from is an advisory drift probe, not part
        # of the seal, so editing the checkout can never make an unchanged
        # installed reaper unhealthy.
        check("reaper health binds private snapshot, unit, and account",
              health["ok"] is True and health["control_ok"] is True
              and health["service_last_result"]["ok"] is True
              and health["stamp"]["control"].get("state_dir") is None
              and "source_files" not in health["stamp"]["control"]
              and str(root) not in stamp_raw
              and {"__init__.py", "cloudlease.py", "campaign.py", "common.py",
                   "jlapi.py", "runpodapi.py", "sshbase.py",
                   "reap_cloud_leases.py"} == source_names == runtime_names
              and expected_public_names == public_runtime_names
              and health["source_drift"] == {
                  "drift": False, "changed": [], "reason": None})
        check("template service is generic and carries no key material",
              "%i" in template_service_text
              and "ExecStart" not in template_service_text
              and "--runpod-key-file" not in template_service_text
              and "PYTHONNOUSERSITE=1" in template_service_text
              and "UMask=0077" in template_service_text)
        check("template timer targets the instance via %%i",
              "%i" in (unit_dir / "fidelity-cloud-reaper@.timer"
                       ).read_text(encoding="utf-8")
              and "fidelity-cloud-reaper@%i.service" in (
                  unit_dir / "fidelity-cloud-reaper@.timer"
                  ).read_text(encoding="utf-8"))
        check("dropin carries the per-instance ExecStart and no secret value",
              control_manifest["command_argv"][:4] == [
                  control_manifest["interpreter"]["path"], "-I", "-S",
                  str(runtime_entry)]
              and str(runtime_entry) in dropin_text
              and str(source_entry) not in dropin_text
              and '"-I"' in dropin_text and '"-S"' in dropin_text
              and "WorkingDirectory=%s\n" % health_dir in dropin_text
              and 'WorkingDirectory="' not in dropin_text
              and "--runpod-key-file" in dropin_text
              and str(root / "key") in dropin_text
              and control_manifest["runtime_root"]
              == str(runtime_entry.parent))
        check("template service carries no key-file path",
              "--runpod-key-file" not in template_service_text
              and "api_key" not in template_service_text)
        check("installer starts immutable oneshot before health is trusted",
              "--user start fidelity-cloud-reaper@runpod.service"
              in systemctl_log.read_text(encoding="utf-8"))
        check("latest failed service invalidates an older healthy stamp",
              latest_service_failed["ok"] is False
              and latest_service_failed["stamp_ok"] is True
              and latest_service_failed["service_last_result"]["result"]
                  == "exit-code"
              and latest_service_failed["service_last_result"]
                  ["exec_main_status"] == "2")
        no_linger = systemd_reaper_health(
            loginctl=str(linger_no), **health_args)
        unavailable_linger = systemd_reaper_health(
            loginctl=str(root / "missing-loginctl"), **health_args)
        check("Linger=no or unavailable loginctl fails health closed",
              no_linger["ok"] is False
              and unavailable_linger["ok"] is False)
        check("timer install refuses without proven login linger",
              raises(LeaseError, lambda: install_systemd_user_timer(
                  ["/bin/true"], lease_dir=lease_health_dir,
                  provider="runpod", provider_account_id="acct-test",
                  state_dir=root / "install-refusal",
                  unit_dir=root / "refusal-units",
                  systemctl=str(systemctl), loginctl=str(linger_no))))
        # The checkout is read once at install and copied into a 0600
        # snapshot inside the 0700 state directory; the snapshot is what the
        # timer executes.  A umask-002 clone is group-writable by default and
        # refusing it forced a chmod of eight files before every reinstall.
        # These cases run under the secure parent: under /tmp they refused
        # for the snapshot's world-writable ancestor and proved nothing.
        install_probe = {
            "lease_dir": lease_health_dir, "provider": "runpod",
            "provider_account_id": "acct-test",
            "systemctl": str(systemctl), "loginctl": str(linger_yes),
        }
        bad_mode_entry = copy_reaper_source_tree(secure_parent / "bad-mode")
        bad_mode_entry.chmod(0o664)
        group_writable = install_systemd_user_timer(
            [sys.executable, str(bad_mode_entry)],
            state_dir=secure_parent / "bad-mode-state",
            unit_dir=secure_parent / "bad-mode-units", **install_probe)
        snapshot_entry = Path(json.loads(
            Path(group_writable["control"]).read_text(encoding="utf-8"))
            ["command_argv"][3])
        check("reaper install copies a group-writable source into a 0600 snapshot",
              snapshot_entry.is_file()
              and stat.S_IMODE(snapshot_entry.stat().st_mode) == 0o600
              and stat.S_IMODE(snapshot_entry.parent.stat().st_mode) == 0o700
              and snapshot_entry.read_bytes() == bad_mode_entry.read_bytes())
        writable_parent = secure_parent / "writable-parent"
        writable_parent.mkdir(mode=0o700)
        writable_entry = copy_reaper_source_tree(writable_parent / "source")
        writable_parent.chmod(0o770)
        check("reaper install tolerates a writable source ancestor",
              Path(install_systemd_user_timer(
                  [sys.executable, str(writable_entry)],
                  state_dir=secure_parent / "writable-parent-state",
                  unit_dir=secure_parent / "writable-parent-units",
                  **install_probe)["control"]).is_file())
        symlink_parent = secure_parent / "source-link"
        symlink_parent.symlink_to(source_entry.parent, target_is_directory=True)
        check("reaper install refuses a symlink source ancestor",
              raises(LeaseError, lambda: install_systemd_user_timer(
                  [sys.executable,
                   str(symlink_parent / "reap_cloud_leases.py")],
                  state_dir=secure_parent / "symlink-state",
                  unit_dir=secure_parent / "symlink-units", **install_probe)))
        exposed_state = secure_parent / "exposed"
        exposed_state.mkdir(mode=0o700)
        exposed_state.chmod(0o770)
        check("reaper install refuses a writable snapshot ancestor",
              raises(LeaseError, lambda: install_systemd_user_timer(
                  [sys.executable, str(source_entry)],
                  state_dir=exposed_state / "state",
                  unit_dir=exposed_state / "units", **install_probe)))
        exposed_state.chmod(0o700)
        stamp_before_drift = stamp_path.read_bytes()
        check("health writer rejects credentials for another account",
              raises(LeaseError, lambda: write_reaper_health(
                  health_dir, healthy_result, lease_dir=lease_health_dir,
                  provider="runpod", provider_account_id="other-account",
                  now=health_now + 2))
              and stamp_path.read_bytes() == stamp_before_drift)
        service = unit_dir / "fidelity-cloud-reaper@.service"
        service_text = service.read_text(encoding="utf-8")
        service.write_text(service_text + "# stale\n", encoding="utf-8")
        check("stale installed template service fails health closed",
              systemd_reaper_health(
                  loginctl=str(linger_yes), **health_args)["ok"] is False)
        service.write_text(service_text, encoding="utf-8")
        timer_unit = unit_dir / "fidelity-cloud-reaper@.timer"
        timer_text = timer_unit.read_text(encoding="utf-8")
        timer_unit.write_text(timer_text + "# stale\n", encoding="utf-8")
        check("stale installed template timer fails health closed",
              systemd_reaper_health(
                  loginctl=str(linger_yes), **health_args)["ok"] is False)
        timer_unit.write_text(timer_text, encoding="utf-8")
        dropin_stale = dropin.read_text(encoding="utf-8")
        dropin.write_text(dropin_stale + "# stale\n", encoding="utf-8")
        check("stale installed dropin fails health closed",
              systemd_reaper_health(
                  loginctl=str(linger_yes), **health_args)["ok"] is False)
        dropin.write_text(dropin_stale, encoding="utf-8")
        source_text = source_entry.read_text(encoding="utf-8")
        source_entry.write_text(source_text + "# source drift\n",
                                encoding="utf-8")
        drifted = systemd_reaper_health(
            loginctl=str(linger_yes), **health_args)
        check("source drift is advisory: health stays ok, snapshot bytes "
              "unchanged, drift names the file",
              drifted["ok"] is True
              and drifted["source_drift"]["drift"] is True
              and drifted["source_drift"]["changed"]
                  == ["bin/reap_cloud_leases.py"]
              and runtime_entry.read_bytes() == installed_entry_bytes
              and str(source_entry) not in dropin_text)
        source_entry.write_text(source_text, encoding="utf-8")
        wrong_state_args = dict(health_args, state_dir=root / "wrong-state")
        check("wrong reaper state directory fails health closed",
              systemd_reaper_health(
                  loginctl=str(linger_yes), **wrong_state_args)["ok"] is False)
        tampered = json.loads(stamp_path.read_text(encoding="utf-8"))
        tampered["completed_at_epoch"] = 1001.0
        stamp_path.write_text(json.dumps(tampered), encoding="utf-8")
        check("tampered health stamp fails closed",
              systemd_reaper_health(
                  loginctl=str(linger_yes), **health_args)["ok"] is False)
        writable_parent.chmod(0o700)
        shutil.rmtree(secure_parent)

        # -- template rendering, per-instance stamp isolation, idempotency --
        template_root = Path(tempfile.mkdtemp(
            prefix="fidelity-reaper-template-", dir=str(Path.home())))
        template_root.chmod(0o700)
        t_health_dir = template_root / "state"
        t_lease_dir = t_health_dir / "leases-v2"
        t_source_parent = template_root / "sources"
        t_source_parent.mkdir(mode=0o700)
        t_unit_dir = template_root / "units"
        t_source_entry = copy_reaper_source_tree(t_source_parent / "trusted")
        t_source_command = [
            sys.executable, str(t_source_entry),
            "--provider", "runpod", "--sweep",
            "--lease-dir", str(t_lease_dir),
            "--reaper-state-dir", str(t_health_dir),
            "--runpod-key-file", str(template_root / "key"),
        ]
        install_result = install_systemd_user_timer(
            t_source_command, lease_dir=t_lease_dir,
            provider="runpod", provider_account_id="acct-isolation",
            state_dir=t_health_dir, unit_dir=t_unit_dir,
            systemctl=str(systemctl), loginctl=str(linger_yes))
        t_template_svc = t_unit_dir / "fidelity-cloud-reaper@.service"
        t_template_tmr = t_unit_dir / "fidelity-cloud-reaper@.timer"
        t_dropin = (t_unit_dir / "fidelity-cloud-reaper@runpod.service.d"
                    / "override.conf")
        check("template rendering: both units exist, use %%i, no key in template",
              t_template_svc.is_file() and t_template_tmr.is_file()
              and t_dropin.is_file()
              and "%i" in t_template_svc.read_text(encoding="utf-8")
              and "%i" in t_template_tmr.read_text(encoding="utf-8")
              and "ExecStart" not in t_template_svc.read_text(
                  encoding="utf-8")
              and "--runpod-key-file" not in t_template_svc.read_text(
                  encoding="utf-8")
              and "--runpod-key-file" in t_dropin.read_text(
                  encoding="utf-8"))
        check("template rendering: control and health are per-provider files",
              (t_health_dir / "reaper-control-runpod.json").is_file()
              and not (t_health_dir / "reaper-control.json").is_file()
              and install_result["control"]
              == str(t_health_dir / "reaper-control-runpod.json")
              and install_result["health_stamp"]
              == str(t_health_dir / "reaper-health-runpod.json"))
        # Write a runpod health stamp using the same monkey-patch
        # technique as the main install test above.
        healthy_result = ReaperResult(
            ok=True, actions=tuple(), failures=tuple(), unresolved=tuple())
        real_verifier = cloudlease_module.verify_reaper_control_account
        cloudlease_module.verify_reaper_control_account = (
            lambda state_dir, **kwargs:
            cloudlease_module._verified_control(
                state_dir, lease_dir=kwargs["lease_dir"],
                provider=kwargs["provider"],
                provider_account_id=kwargs["provider_account_id"]))
        try:
            t_stamp_path = write_reaper_health(
                t_health_dir, healthy_result, lease_dir=t_lease_dir,
                provider="runpod", provider_account_id="acct-isolation",
                now=time.time())
        finally:
            cloudlease_module.verify_reaper_control_account = real_verifier
        check("per-instance stamp isolation: runpod stamp is its own file",
              t_stamp_path.is_file()
              and t_stamp_path == t_health_dir / "reaper-health-runpod.json"
              and not (t_health_dir / "reaper-health.json").is_file()
              and not (t_health_dir / "reaper-health-vast.json").is_file())
        # A stale stamp for another provider must not block runpod health.
        cloudlease_module._atomic_replace(
            t_health_dir / "reaper-health-vast.json",
            {"schema": HEALTH_SCHEMA, "ok": False,
             "invocation_id": "0" * 32,
             "invocation_started_at_epoch": 1.0,
             "invocation_started_at_utc": "1970-01-01T00:00:01Z",
             "completed_at_epoch": 1.0,
             "completed_at_utc": "1970-01-01T00:00:01Z",
             "control": {}, "actions": [], "failure_count": 1,
             "unresolved_count": 0, "result_sha256": "x"})
        t_health = systemd_reaper_health(
            state_dir=t_health_dir, lease_dir=t_lease_dir,
            provider="runpod", provider_account_id="acct-isolation",
            systemctl=str(systemctl), loginctl=str(linger_yes),
            now=time.time(), max_age_seconds=900)
        check("stale stamp for another provider does not block runpod",
              t_health["ok"] is True)

        # Installer idempotency: re-install does not duplicate drop-ins.
        # A re-install creates a new runtime snapshot (different digest),
        # so the dropin content changes — but the file is replaced, not
        # duplicated.
        install_systemd_user_timer(
            t_source_command, lease_dir=t_lease_dir,
            provider="runpod", provider_account_id="acct-isolation",
            state_dir=t_health_dir, unit_dir=t_unit_dir,
            systemctl=str(systemctl), loginctl=str(linger_yes))
        check("re-install is idempotent: dropin not duplicated",
              t_dropin.is_file()
              and len(list((t_unit_dir
                            / "fidelity-cloud-reaper@runpod.service.d"
                           ).iterdir())) == 1)
        check("re-install is idempotent: one control file per provider",
              (t_health_dir / "reaper-control-runpod.json").is_file()
              and len([p for p in t_health_dir.iterdir()
                       if p.name.startswith("reaper-control-")]) == 1)
        shutil.rmtree(template_root)

        termination = 2000000000
        observation = (
            termination + MAX_PROVIDER_DEADLINE_OBSERVATION_LAG_SECONDS)
        drill_request = {
            "drill_mode": PROVIDER_DEADLINE_DRILL_MODE,
            "provider_account_id": "acct-test",
            "campaign_ledger": None, "campaign_attempt_key": None,
            "secure_cloud": True, "offer": "on-demand", "spot": False,
            "gpu_type_id": "NVIDIA L4", "gpu_count": 1,
            "image_name": "runpod/pytorch", "volume_gb": 100,
            "container_disk_gb": 100, "min_vcpu": 4, "min_ram_gb": 16,
            "network_volume_id": None,
            "terminate_after": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(termination)),
            "provider_deadline_observation_until": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(observation)),
            "pre_create_safety": server_time_fixture(),
            "prepared_create": prepared_create_fixture(
                name="fidcloud-%s-a%s" % ("9" * 64, "b" * 24),
                gpu="NVIDIA L4", count=1, volume_gb=100,
                container_disk_gb=100, min_vcpu=4, min_ram_gb=16,
                image="runpod/pytorch",
                terminate_after=time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(termination))),
            "producer_checkout": {
                "schema": "fidelity-suite/producer-checkout.v1",
                "revision": "1" * 40,
                "initial": {
                    "untracked_files": "all",
                    "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
                    "status_bytes": 0,
                    "clean": True,
                },
                "pre_post": {
                    "untracked_files": "all",
                    "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
                    "status_bytes": 0,
                    "clean": True,
                },
            },
        }
        drill_store = LeaseStore(
            root / "provider-deadline", clock=lambda: 1999999000.0)
        drill = drill_store.begin_create(
            job_hash="9" * 64, provider="runpod", request=drill_request,
            pre_create_resources=[],
            attempt_id="b" * 24,
            create_deadline_epoch=1999999060,
            workload_deadline_epoch=termination - 10)
        drill = drill_store.record_post_intent(drill)
        drill_name = drill_store.read(drill)["create"]["exact_name"]
        drill = drill_store.record_create_success(
            drill, {"id": "pod-drill", "name": drill_name})
        drill_provider = StatefulProvider([
            {"id": "pod-drill", "name": drill_name, "status": "RUNNING"}])
        before_deadline = reap_once(
            drill_store, {"runpod": drill_provider},
            now=termination - 1)
        check("provider deadline drill preserves pod before reap deadline",
              drill_provider.destroyed == []
              and drill_store.read(drill)["state"] == ACTIVE
              and not before_deadline.actions)
        expired = reap_once(
            drill_store, {"runpod": drill_provider}, now=termination)
        check("provider deadline drill independently destroys at reap deadline",
              drill_provider.destroyed == ["pod-drill"]
              and any(action["action"] == "destroy-requested"
                      and action["provider_id"] == "pod-drill"
                      for action in expired.actions)
              and not expired.unresolved)

        bad_request = dict(drill_request)
        bad_request["provider_deadline_observation_until"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(
                termination
                + MAX_PROVIDER_DEADLINE_OBSERVATION_LAG_SECONDS + 1))
        bad_store = LeaseStore(
            root / "bad-provider-deadline", clock=lambda: 1999999000.0)
        check("overlong provider deadline observation lag fails closed",
              raises(LeaseError, lambda: bad_store.begin_create(
                  job_hash="8" * 64, provider="runpod",
                  request=bad_request, pre_create_resources=[],
                  create_deadline_epoch=1999999060,
                  workload_deadline_epoch=termination - 10)))


class RecordingRunPod(RunPod):
    def __init__(self, dry=False):
        super().__init__(dry=dry, key_file="/not/read")
        self._key = "fixture-runpod-key"
        self.queries = []

    def _validated_ssh_public_key(self):
        return "ssh-ed25519 AAAA arbitrary comment"

    def prepare_safe_create(self, **kw):
        prepared = super().prepare_safe_create(**kw)
        owner = self

        class FixtureResponse:
            status = 200

            class Headers:
                @staticmethod
                def get_content_type():
                    return "application/json"

                @staticmethod
                def get(name):
                    return None

            headers = Headers()

            def __init__(self, body):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            @staticmethod
            def geturl():
                return "https://api.runpod.io/graphql"

            @staticmethod
            def getcode():
                return 200

            def read(self, unused_limit=None):
                return self.body

        class FixtureOpener:
            @staticmethod
            def open(request, timeout=180):
                query = json.loads(request.data.decode("utf-8"))["query"]
                data = owner._gql(query, timeout=timeout)
                return FixtureResponse(json.dumps({"data": data}).encode("utf-8"))

        return replace(prepared, http_opener=FixtureOpener())

    def _gql(self, query, *, timeout=60):
        self.queries.append(query)
        return {"podFindAndDeployOnDemand": {
            "id": "pod-created", "name": "exact-name", "costPerHr": "1.25"}}


class ListingRunPod(RecordingRunPod):
    def _pods(self):
        return [{
            "id": "pod-exited", "name": "exact-name", "desiredStatus": "EXITED",
            "costPerHr": "1.25", "runtime": {"uptimeInSeconds": 4, "ports": []},
            "gpuCount": 1, "volumeInGb": 100, "containerDiskInGb": 20,
            "networkVolumeId": None, "imageName": "image",
            "terminateAfter": "2030-01-02T03:04:05Z",
            "machine": {
                "id": "host-1", "gpuTypeId": "A100",
                "gpuDisplayName": "NVIDIA A100", "secureCloud": True,
                "currentPricePerGpu": "1.25", "podHostId": "host-1",
            },
        }]
class TransitionListingRunPod(RecordingRunPod):
    def _gql(self, query, *, timeout=60):
        return {"myself": {"pods": [{
            "id": "pod-created", "name": "created-name",
            "desiredStatus": "CREATED", "costPerHr": None,
            "runtime": None, "gpuCount": 1, "volumeInGb": 100.0,
            "containerDiskInGb": 20, "networkVolumeId": None,
            "imageName": "image", "machine": None,
        }, {
            "id": "pod-exited-null-machine", "name": "exited-name",
            "desiredStatus": "EXITED", "costPerHr": 0,
            "runtime": {"uptimeInSeconds": 4.0, "ports": []},
            "gpuCount": 1.0, "volumeInGb": "20.0",
            "containerDiskInGb": 20.0, "networkVolumeId": None,
            "imageName": "image", "machine": None,
        }]}}




class BillingRunPod(RecordingRunPod):
    def __init__(self, response):
        super().__init__()
        self.response = response
        self.billing_query = None

    def _get_v2(self, path, query, *, timeout=60):
        self.billing_query = (path, dict(query))
        return self.response

class InventoryRunPod(RecordingRunPod):
    def __init__(self, volume_outage=False):
        super().__init__()
        self.volume_outage = volume_outage

    def _get_v1(self, path, query=None, *, timeout=60):
        if path == "/pods":
            return [{
                "id": "pod-with-volume", "name": "old-pod",
                "desiredStatus": "EXITED", "costPerHr": "0.50",
                "adjustedCostPerHr": "0.45",
                "networkVolume": {
                    "id": "volume-1", "name": "persistent", "size": 100},
            }]
        if self.volume_outage:
            raise RunPodError("volume inventory unavailable")
        return [{"id": "volume-1", "name": "persistent",
                 "size": 100, "dataCenterId": "US-KS-2"}]

class MissingPodsRunPod(RecordingRunPod):
    def _gql(self, query, *, timeout=60):
        return {}

class BidOnlyRunPod(RecordingRunPod):
    def _gql(self, query, *, timeout=60):
        if "lowestPrice" not in query:
            return {"gpuTypes": [{
                "id": "A100", "displayName": "A100", "memoryInGb": 80,
                "communityCloud": False, "secureCloud": True,
            }]}
        return {"gpuTypes": [{"lowestPrice": {
            "minimumBidPrice": "0.25",
            "uninterruptablePrice": None,
            "stockStatus": "High",
        }}]}


class MultiBillingRunPod(RecordingRunPod):
    def __init__(self):
        super().__init__()
        self.asked = []

    def _get_v2(self, path, query, *, timeout=60):
        self.asked.append(query["podId"])
        total = {"pod-a": "1.10", "pod-b": "2.20"}[query["podId"]]
        amounts = {
            "totalAmount": total, "gpuAmount": total,
            "cpuAmount": "0.00", "diskAmount": "0.00",
        }
        start_epoch = calendar.timegm(time.strptime(
            query["startTime"], "%Y-%m-%dT%H:%M:%SZ"))
        end_epoch = calendar.timegm(time.strptime(
            query["endTime"], "%Y-%m-%dT%H:%M:%SZ"))
        resolved_start_epoch = start_epoch - start_epoch % 3600
        resolved_end_epoch = ((end_epoch + 3599) // 3600) * 3600
        resolved_query = dict(
            query,
            startTime=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(resolved_start_epoch)),
            endTime=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(resolved_end_epoch)))
        zero_amounts = {
            "totalAmount": "0", "gpuAmount": "0",
            "cpuAmount": "0", "diskAmount": "0",
        }
        records = []
        for bucket_start in range(
                resolved_start_epoch, resolved_end_epoch, 3600):
            records.append(dict(
                amounts if bucket_start == resolved_start_epoch
                else zero_amounts,
                podId=query["podId"],
                startTime=time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(bucket_start)),
                endTime=time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(bucket_start + 3600))))
        return {
            "records": records,
            "metadata": {
                "query": resolved_query, "recordCount": len(records),
                "uniquePodCount": 1, "totals": amounts},
        }


class MissingSSHRunPod(RunPod):
    def __init__(self, ssh_key):
        super().__init__(dry=False, key_file="/not/read", ssh_key=ssh_key)
        self.provider_calls = 0

    def _gql(self, query, *, timeout=60):
        self.provider_calls += 1
        return {}

def runpod_cases():
    print("\n== RunPod key, deadline, mount, list, and billing contracts ==")
    check("RunPod JSON rejects duplicate financial fields",
          raises(RunPodError, lambda: _strict_json_loads(
              '{"data":{"costPerHr":"1","costPerHr":"2"}}')))
    check("RunPod JSON rejects NaN and Infinity tokens",
          raises(RunPodError, lambda: _strict_json_loads(
              '{"clientBalance":NaN}'))
          and raises(RunPodError, lambda: _strict_json_loads(
              '{"currentSpendPerHr":Infinity}')))
    malformed_status = RecordingRunPod()
    malformed_status._gql = lambda query, timeout=60: {
        "myself": {"id": "acct", "clientBalance": "NaN",
                   "currentSpendPerHr": "0"}}
    check("RunPod status rejects non-finite account money",
          raises(RunPodError, malformed_status.status))
    timestamped_status = RecordingRunPod()
    timestamped_status._gql = lambda query, timeout=60: {
        "myself": {"id": "acct", "clientBalance": "10.5",
                   "currentSpendPerHr": "0"}}
    status_document = timestamped_status.status()
    status_observed = status_document.get("observed_at_utc")
    check("RunPod status binds a canonical controller receipt time",
          set(status_document) == {
              "id", "clientBalance", "currentSpendPerHr", "observed_at_utc"}
          and isinstance(status_observed, str)
          and time.strftime(
              "%Y-%m-%dT%H:%M:%SZ",
              time.strptime(status_observed, "%Y-%m-%dT%H:%M:%SZ"))
          == status_observed)
    injection_provider = RecordingRunPod()
    injection_calls = len(injection_provider.queries)
    check("injection-shaped pod id is refused before mutation construction",
          raises(RunPodError, lambda: injection_provider.destroy(
              'pod"})} mutation { podTerminate(input:{podId:"victim'))
          and len(injection_provider.queries) == injection_calls)
    check("injection-shaped GPU id is refused before create construction",
          raises(RunPodError, lambda: RecordingRunPod(dry=True).create(
              gpu_type='A100"})} mutation { podTerminate',
              name="exact-name", region="secure", storage_gb=100,
              container_disk_gb=20,
              terminate_after="2030-01-02T03:04:05Z")))
    hostile_catalog = RecordingRunPod()
    catalog_calls = []
    hostile_catalog._gql = lambda query, timeout=60: (
        catalog_calls.append(query) or {"gpuTypes": [{
            "id": 'A100"}) { mutation', "displayName": "bad",
            "memoryInGb": 80, "communityCloud": False,
            "secureCloud": True}]})
    check("injection-shaped catalog GPU id gets no price query",
          raises(RunPodError, hostile_catalog.gpus)
          and len(catalog_calls) == 1)
    filtered_catalog = RecordingRunPod()
    filtered_queries = []
    def filtered_catalog_query(query, timeout=60):
        filtered_queries.append(query)
        if "lowestPrice" not in query:
            return {"gpuTypes": [
                {"id": "NVIDIA H200", "displayName": "H200",
                 "memoryInGb": 141, "communityCloud": False,
                 "secureCloud": True},
                {"id": "NVIDIA L4", "displayName": "L4",
                 "memoryInGb": 24, "communityCloud": False,
                 "secureCloud": True},
            ]}
        return {"gpuTypes": [{"lowestPrice": {
            "minimumBidPrice": "0.49",
            "uninterruptablePrice": "0.49",
            "stockStatus": "Low",
        }}]}
    filtered_catalog._gql = filtered_catalog_query
    filtered_offers = filtered_catalog.gpus(
        gpu_type="NVIDIA L4", secure_only=True)
    check("exact GPU filter avoids unrelated catalogue price queries",
          len(filtered_queries) == 2
          and "NVIDIA L4" in filtered_queries[1]
          and "NVIDIA H200" not in filtered_queries[1]
          and len(filtered_offers) == 1
          and filtered_offers[0].gpu_type == "NVIDIA L4"
          and filtered_offers[0].region == "secure")

    class HostileHeaders:
        @staticmethod
        def get_content_type():
            return "application/json"

        @staticmethod
        def get(name):
            if name == "Date":
                return "Mon, 01 Jan 2024 00:00:00 GMT"
            return None

    class HostileRedirectResponse:
        status = 200
        headers = HostileHeaders()

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

        @staticmethod
        def geturl():
            return "https://attacker.invalid/stolen"

        @staticmethod
        def getcode():
            return 200

        @staticmethod
        def read(unused_limit=None):
            return b'{"data":{"myself":{"id":"acct"}}}'

    original_urlopen = runpod_module.safe_urlopen
    redirected = RunPod(key_file="/not/read")
    redirected._key = "runpod-hostile-redirect-secret"
    redirect_error = ""
    try:
        runpod_module.safe_urlopen = (
            lambda request, timeout=60: HostileRedirectResponse())
        redirected.status()
    except RunPodError as exc:
        redirect_error = str(exc)
    finally:
        runpod_module.safe_urlopen = original_urlopen
    check("cross-origin RunPod redirect fails closed without key disclosure",
          "crossed its HTTPS origin" in redirect_error
          and redirected._key not in redirect_error)

    # A single provider 503 on a read-only poll aborted a paid drill whose pod
    # was already destroyed and whose absence was already proven. Reads must
    # ride out a bounded outage; mutations must never be repeated.
    class OutageResponse:
        status = 200
        url = runpod_module.GQL
        headers = email.message.Message()
        headers.add_header("Content-Type", "application/json")
        headers.add_header(
            "Date",
            time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()))

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

        def geturl(self):
            return runpod_module.GQL

        def getcode(self):
            return 200

        def info(self):
            return {}

        def read(self, unused_limit=None):
            return (b'{"data":{"myself":{"id":"acct",'
                    b'"clientBalance":"10.5","currentSpendPerHr":"0"}}}')

    def outage_opener(failures, record):
        def opener(request, timeout=60):
            record.append(bytes(request.data))
            if len(record) <= failures:
                raise urllib.error.HTTPError(
                    request.full_url, 503, "Service Unavailable", {},
                    io.BytesIO(b'{"message":"Service Unavailable"}'))
            return OutageResponse()
        return opener

    original_backoff = runpod_module._READ_RETRY_BACKOFF_SECONDS
    recovered_calls = []
    exhausted_calls = []
    mutation_calls = []
    exhausted_error = ""
    mutation_error = ""
    try:
        runpod_module._READ_RETRY_BACKOFF_SECONDS = (0.0, 0.0)
        recovering = RunPod(key_file="/not/read")
        recovering._key = "runpod-outage-secret"
        runpod_module.safe_urlopen = outage_opener(2, recovered_calls)
        recovered_status = recovering.status()

        exhausting = RunPod(key_file="/not/read")
        exhausting._key = "runpod-outage-secret"
        runpod_module.safe_urlopen = outage_opener(99, exhausted_calls)
        try:
            exhausting.status()
        except RunPodError as exc:
            exhausted_error = str(exc)

        mutating = RunPod(key_file="/not/read")
        mutating._key = "runpod-outage-secret"
        runpod_module.safe_urlopen = outage_opener(99, mutation_calls)
        try:
            mutating._gql("mutation { podTerminate(input:{podId:\"p\"}) }")
        except RunPodError as exc:
            mutation_error = str(exc)
    finally:
        runpod_module.safe_urlopen = original_urlopen
        runpod_module._READ_RETRY_BACKOFF_SECONDS = original_backoff
    check("bounded transient outage is ridden out on reads, never mutations",
          recovered_status["id"] == "acct"
          and len(recovered_calls) == 3
          and len(exhausted_calls) == 3
          and "503" in exhausted_error
          and "3 attempt(s)" in exhausted_error
          and len(mutation_calls) == 1
          and "HTTP 503" in mutation_error
          and "attempt(s)" not in mutation_error
          and all(secret not in exhausted_error + mutation_error
                  for secret in ("runpod-outage-secret",)))
    skewed = RunPod(key_file="/not/read")
    now_epoch = time.time()
    skewed._server_time = {
        "schema": "fidelity-suite/runpod-server-time.v1",
        "endpoint_origin": "https://api.runpod.io",
        "date_header": "Mon, 01 Jan 2024 00:00:00 GMT",
        "server_epoch": now_epoch - 31,
        "local_received_epoch": now_epoch,
        "local_minus_server_seconds": 31,
    }
    check("RunPod pre-create server-time evidence refuses clock skew",
          raises(RunPodError, skewed.server_time_evidence))
    skewed._server_time["local_minus_server_seconds"] = 0
    skewed._server_time["local_received_epoch"] = now_epoch - 31
    check("RunPod pre-create server-time evidence refuses stale Date",
          raises(RunPodError, skewed.server_time_evidence))
    check("RunPod sole default key path is stable and absolute",
          os.path.isabs(DEFAULT_KEY_FILE)
          and DEFAULT_KEY_FILE.endswith("/.config/runpod/api_key"))
    with tempfile.TemporaryDirectory() as td:
        key = Path(td) / "runpod.key"
        key.write_text("secret-value\n", encoding="utf-8")
        key.chmod(0o600)
        check("absolute owner-only 0600 key file loads", _load_key(str(key)) == "secret-value")
        key.chmod(0o644)
        check("non-0600 key file is refused",
              raises(RunPodError, lambda: _load_key(str(key))))
        key.chmod(0o600)
        key_link = Path(td) / "runpod-link"
        os.symlink(str(key), str(key_link))
        check("symlinked API key file is refused",
              raises(RunPodError, lambda: _load_key(str(key_link))))
        check("relative key path is refused",
              raises(RunPodError, lambda: _load_key("runpod.key")))

    redirect_prepared_provider = RecordingRunPod()
    redirect_prepared = redirect_prepared_provider.prepare_safe_create(
        gpu_type="A100", name="exact-name", region="secure",
        storage_gb=100, container_disk_gb=20,
        terminate_after="2030-01-02T03:04:05Z")
    redirect_calls = []

    class RedirectRefusalOpener:
        @staticmethod
        def open(request, timeout=180):
            redirect_calls.append(bytes(request.data))
            raise urllib.error.HTTPError(
                request.full_url, 307, "redirect refused", {},
                io.BytesIO(b"redirect"))

    redirect_prepared = replace(
        redirect_prepared, http_opener=RedirectRefusalOpener())
    check("prepared create refuses redirects without resending mutation body",
          raises(
              RunPodError,
              lambda: redirect_prepared_provider.submit_prepared_create(
                  redirect_prepared))
          and redirect_calls == [redirect_prepared.graphql_body]
          and redirect_prepared.to_dict()["graphql_body_sha256"]
          == hashlib.sha256(redirect_prepared.graphql_body).hexdigest())
    provider = RecordingRunPod()
    made = provider.create(
        gpu_type="A100", name="exact-name", region="secure",
        storage_gb=100, container_disk_gb=20,
        terminate_after="2030-01-02T03:04:05Z")
    check("SSH GraphQL create binds deadline, disks, and canonical comment-free key",
          'terminateAfter:"2030-01-02T03:04:05Z"' in provider.queries[-1]
          and "volumeInGb:100, containerDiskInGb:20" in provider.queries[-1]
          and 'value:"ssh-ed25519 AAAA"'
          in provider.queries[-1]
          and "arbitrary comment" not in provider.queries[-1]
          and made["storage_gb"] == 100
          and made["container_disk_gb"] == 20)
    dry_request = RecordingRunPod(dry=True).create(
        gpu_type="A100", name="exact-name", region="secure",
        storage_gb=100, container_disk_gb=20,
        terminate_after="2030-01-02T03:04:05Z")
    check("dry create validates and returns the exact live request identity",
          dry_request["dry_run"] is True
          and dry_request["request"] == made["request"])
    safe_create = {
        "gpu_type": "A100", "name": "exact-name", "region": "secure",
        "storage_gb": 100, "container_disk_gb": 20,
        "terminate_after": "2030-01-02T03:04:05Z",
    }
    def create_response_provider(name, cost):
        candidate = RecordingRunPod()
        candidate._gql = lambda query, timeout=60: {
            "podFindAndDeployOnDemand": {
                "id": "pod-created", "name": name, "costPerHr": cost}}
        return candidate

    null_cost = create_response_provider("exact-name", None).create(**safe_create)
    zero_cost = create_response_provider("exact-name", 0).create(**safe_create)
    malformed_cost = create_response_provider(
        "exact-name", "Infinity").create(**safe_create)
    null_name = create_response_provider(None, None).create(**safe_create)
    wrong_name_create = create_response_provider("other-name", "1.25")
    structured_create_errors = []
    try:
        wrong_name_create.create(**safe_create)
    except RunPodCreateResponseError as exc:
        structured_create_errors.append(exc)
    check("exact create id acknowledges lagging response economics and name",
          null_cost["pod_id"] == "pod-created"
          and null_cost["cost_per_hr"] is None
          and zero_cost["cost_per_hr"] == 0
          and malformed_cost["cost_per_hr"] == "Infinity"
          and null_name["name"] is None
          and null_name["cost_per_hr"] is None)
    with tempfile.TemporaryDirectory() as lease_td:
        response_store = LeaseStore(Path(lease_td))
        response_ref = begin(response_store, "d")
        response_ref = response_store.record_create_success(
            response_ref, null_name)
        response_evidence = response_store.read(
            response_ref)["history"][-1]["evidence"]["response"]
    check("lease preserves nullable create response without inferring name",
          response_evidence["name"] is None
          and response_evidence["cost_per_hr"] is None
          and response_evidence["name_matches_exact"] is False)
    check("exact nonempty response-name mismatch alone retains cleanup id",
          len(structured_create_errors) == 1
          and structured_create_errors[0].provider_id == "pod-created"
          and structured_create_errors[0].response["id"] == "pod-created")
    complete_row = json.loads(json.dumps(ListingRunPod()._pods()[0]))
    complete_row["id"] = "pod-created"
    complete_row["desiredStatus"] = "RUNNING"
    convergence_provider = ListingRunPod()
    convergence_provider._pods = lambda: [complete_row]
    converged = convergence_provider.validate_safe_resource_binding(
        "pod-created", expected_name="exact-name", gpu_type_id="A100",
        secure_cloud=True, gpu_count=1, volume_gb=100,
        container_disk_gb=20, image_name="image",
        terminate_after="2030-01-02T03:04:05Z")
    check("null response cost/name can qualify only after complete live listing",
          converged["passed"] is True
          and converged["observed"]["name"] == "exact-name"
          and converged["observed"]["cost_per_hr"] == "1.25")
    attester = RecordingRunPod()
    expected_vram = 24 * 1024 * 1024 * 1024
    remote_now = int(time.time())
    attester.exec_stdout = lambda machine_id, command, timeout=600: json.dumps({
        "remote_time_epoch": remote_now,
        "remote_time_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(remote_now)),
        "logical_cpus": 8,
        "memtotal_bytes": 64 * 1024 * 1024 * 1024,
        "effective_memory_bytes": 60 * 1024 * 1024 * 1024,
        "nvidia_smi_exit_code": 0,
        "nvidia_smi_error": "",
        "gpus": [{
            "index": 0, "name": "NVIDIA L4",
            "vram_bytes": expected_vram,
            "driver_version": "575.57",
        }],
        "cuda": {
            "usable": True, "count": 1, "name": "NVIDIA L4",
            "vram_bytes": expected_vram, "error": None, "interpreter": "python3",
        },
        "filesystems": {
            "container": {
                "path": "/", "mount_point": "/", "fs_type": "overlay",
                "source": "overlay", "device": 1,
                "total_bytes": 25_000_000_000,
                "available_bytes": 20_000_000_000,
            },
            "workspace": {
                "path": "/workspace", "mount_point": "/workspace",
                "fs_type": "ext4", "source": "/dev/volume", "device": 2,
                "total_bytes": 110_000_000_000,
                "available_bytes": 100_000_000_000,
            },
        },
    })
    attestation = attester.attest_live_resource(
        "pod-created", expected_gpu_model="NVIDIA L4",
        expected_vram_bytes=expected_vram, min_vcpu=8, min_ram_gb=60,
        volume_gb=100, container_disk_gb=20,
        workspace_available_bytes_minimum=90_000_000_000,
        container_available_bytes_minimum=15_000_000_000)
    attestation_body = dict(attestation)
    attestation_digest = attestation_body.pop("attestation_sha256")
    check("SSH live resource attestation binds hardware CUDA and filesystems",
          attestation["ok"] is True
          and all(attestation["checks"].values())
          and attestation_digest == hashlib.sha256(json.dumps(
              attestation_body, sort_keys=True, separators=(",", ":"),
              ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest())
    stale_attester = RecordingRunPod()
    stale_observed = json.loads(attester.exec_stdout(
        "pod-created", "unused"))
    stale_observed["remote_time_epoch"] -= 3600
    stale_observed["remote_time_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(stale_observed["remote_time_epoch"]))
    stale_attester.exec_stdout = (
        lambda machine_id, command, timeout=600:
        json.dumps(stale_observed))
    stale_attestation = stale_attester.attest_live_resource(
        "pod-created", expected_gpu_model="NVIDIA L4",
        expected_vram_bytes=expected_vram, min_vcpu=8, min_ram_gb=60,
        volume_gb=100, container_disk_gb=20,
        workspace_available_bytes_minimum=90_000_000_000,
        container_available_bytes_minimum=15_000_000_000)
    check("SSH live resource attestation refuses a stale pod clock",
          stale_attestation["ok"] is False
          and stale_attestation["checks"]["remote_clock"] is False
          and "remote_clock" in stale_attestation["failures"])
    low_free_checks = []
    for role, available in (
            ("workspace", 89_999_999_999),
            ("container", 14_999_999_999)):
        low_observed = json.loads(attester.exec_stdout(
            "pod-created", "unused"))
        low_observed["filesystems"][role]["available_bytes"] = available
        low_attester = RecordingRunPod()
        low_attester.exec_stdout = (
            lambda machine_id, command, timeout=600, row=low_observed:
            json.dumps(row))
        low_attestation = low_attester.attest_live_resource(
            "pod-created", expected_gpu_model="NVIDIA L4",
            expected_vram_bytes=expected_vram, min_vcpu=8, min_ram_gb=60,
            volume_gb=100, container_disk_gb=20,
            workspace_available_bytes_minimum=90_000_000_000,
            container_available_bytes_minimum=15_000_000_000)
        check_name = "%s_available_bytes" % role
        low_free_checks.append(
            low_attestation["ok"] is False
            and low_attestation["checks"][check_name] is False
            and check_name in low_attestation["failures"])
    check("SSH live attestation refuses low workspace and container free bytes",
          all(low_free_checks))
    check("dry create refuses community and spot capacity",
          raises(RunPodError, lambda: RecordingRunPod(dry=True).create(
              **dict(safe_create, region="community")))
          and raises(RunPodError, lambda: RecordingRunPod(dry=True).create(
              **dict(safe_create, spot=True)))
          and raises(RunPodError, lambda: RecordingRunPod(dry=True).create(
              **dict(safe_create, offer="spot"))))
    soon = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() + MIN_CREATE_SETUP_SECONDS - 30))
    check("dry create refuses past or insufficient terminateAfter margin",
          raises(RunPodError, lambda: RecordingRunPod(dry=True).create(
              **dict(safe_create, terminate_after="2000-01-01T00:00:00Z")))
          and raises(RunPodError, lambda: RecordingRunPod(dry=True).create(
              **dict(safe_create, terminate_after=soon))))
    call_count = len(provider.queries)
    control_key = RecordingRunPod(dry=True)
    control_key._validated_ssh_public_key = (
        lambda: "ssh-ed25519 AAAA\ninjected")
    check("dry create rejects multi-line injected public key",
          raises(RunPodError, lambda: control_key.create(**safe_create)))
    check("network volume is refused before any provider call",
          raises(RunPodError, lambda: provider.create(
              gpu_type="A100", terminate_after="2030-01-02T03:04:05Z",
              network_volume_id="volume-1"))
          and len(provider.queries) == call_count)
    check("native docker launch is refused before any provider call",
          raises(RunPodError, lambda: provider.create(
              gpu_type="A100", terminate_after="2030-01-02T03:04:05Z",
              docker_cmd=["python", "run.py"]))
          and len(provider.queries) == call_count)
    check("non-canonical deadline is refused",
          raises(RunPodError, lambda: provider.create(
              gpu_type="A100", terminate_after="2030-01-02T03:04:05+00:00")))
    check("missing explicit container disk size is refused",
          raises(RunPodError, lambda: RunPod(dry=True).create(
              gpu_type="A100", storage_gb=100)))
    check("bid-only capacity is not quoted as safe on-demand capacity",
          BidOnlyRunPod().gpus() == [])
    check("garbage region is refused rather than silently mapped to secure",
          raises(RunPodError, lambda: RecordingRunPod(dry=True).create(
              gpu_type="A100", name="exact-name", region="moon",
              storage_gb=100, container_disk_gb=20,
              terminate_after="2030-01-02T03:04:05Z")))
    with tempfile.TemporaryDirectory() as key_td:
        missing_key = MissingSSHRunPod(str(Path(key_td) / "missing"))
        check("missing SSH key pair refuses before a provider POST",
              raises(RunPodError, lambda: missing_key.create(
                  gpu_type="A100", name="exact-name", region="secure",
                  storage_gb=100, container_disk_gb=20,
                  terminate_after="2030-01-02T03:04:05Z"))
              and missing_key.provider_calls == 0)
    check("pod identity query requests authoritative GPU/cloud/disk/mount fields",
          all(field in RunPod._POD_FIELDS for field in (
              "containerDiskInGb", "volumeInGb", "networkVolumeId",
              "gpuTypeId", "gpuDisplayName", "secureCloud",
              "currentPricePerGpu", "podHostId")))

    listed = ListingRunPod().list_lifecycle_resources()
    check("lifecycle listing exposes exact id and EXITED status without hiding it",
          listed[0]["id"] == "pod-exited"
          and listed[0]["status"] == "EXITED"
          and listed[0]["listed"] is True)
    transition_provider = TransitionListingRunPod()
    transition_instances = transition_provider.list_instances()
    transition_rows = transition_provider.list_lifecycle_resources()
    check("CREATED/EXITED rows retain exact cleanup identity with null economics",
          [str(item.machine_id) for item in transition_instances]
          == ["pod-created", "pod-exited-null-machine"]
          and [item["status"] for item in transition_rows]
          == ["CREATED", "EXITED"]
          and transition_rows[0]["cost_per_hr"] is None
          and transition_rows[1]["cost_per_hr"] == "0")
    check("GraphQL Float disk fields normalize only when exactly integral",
          transition_rows[0]["volume_gb"] == 100
          and transition_rows[1]["volume_gb"] == 20
          and transition_rows[1]["container_disk_gb"] == 20)
    check("nullable listing identity still refuses scientific post-create binding",
          raises(RunPodError, lambda:
                 transition_provider.validate_safe_resource_binding(
                     "pod-created", expected_name="created-name",
                     gpu_type_id="A100", secure_cloud=True, gpu_count=1,
                     volume_gb=100, container_disk_gb=20,
                     image_name="image",
                     terminate_after="2030-01-02T03:04:05Z")))
    check("missing myself.pods is unknown, never a complete empty listing",
          raises(RunPodError, lambda: MissingPodsRunPod().list_instances()))
    listing_provider = ListingRunPod()
    check("resource detail accepts exact id and never controls by name",
          listing_provider.get("pod-exited") is not None
          and listing_provider.get("exact-name") is None)
    check("pause and recovery are refused even in dry mode",
          raises(RunPodError, lambda: RunPod(dry=True).pause("pod-exited"))
          and raises(RunPodError,
                     lambda: RunPod(dry=True).resume("pod-exited")))
    binding = listing_provider.validate_safe_resource_binding(
        "pod-exited", expected_name="exact-name", gpu_type_id="A100",
        secure_cloud=True, gpu_count=1, volume_gb=100,
        container_disk_gb=20, image_name="image",
        terminate_after="2030-01-02T03:04:05Z")
    check("post-create binding verifies GPU/cloud/disks/image/name/no mount/deadline",
          binding["passed"] is True
          and binding["terminate_after_observable"] is True
          and listing_provider.get("pod-exited").gpu_type == "A100")
    check("post-create binding refuses wrong GPU identity",
          raises(RunPodError, lambda:
                 listing_provider.validate_safe_resource_binding(
                     "pod-exited", expected_name="exact-name",
                     gpu_type_id="H100", secure_cloud=True, gpu_count=1,
                     volume_gb=100, container_disk_gb=20,
                     image_name="image",
                     terminate_after="2030-01-02T03:04:05Z")))

    inventory = InventoryRunPod().chargeable_inventory()
    check("chargeable inventory includes attached volume and persistent volume",
          inventory["complete"] is True
          and inventory["families"]["pods"]["resources"][0]
          ["network_volume_id"] == "volume-1"
          and inventory["families"]["network_volumes"]["resources"][0]["id"]
          == "volume-1")
    incomplete = InventoryRunPod(volume_outage=True).chargeable_inventory()
    check("unavailable volume enumeration is explicit unknown, never empty",
          incomplete["complete"] is False
          and incomplete["unknown_families"] == ["network_volumes"]
          and incomplete["families"]["network_volumes"]["complete"] is False)
    bad_volume = InventoryRunPod()
    bad_volume._get_v1 = lambda path, query=None, timeout=60: [
        {"id": "volume-1", "size": "NaN", "costPerHr": "0.01"}]
    duplicate_volume = InventoryRunPod()
    duplicate_volume._get_v1 = lambda path, query=None, timeout=60: [
        {"id": "volume-1", "size": 100},
        {"id": "volume-1", "size": 100}]
    bad_volume_rate = InventoryRunPod()
    bad_volume_rate._get_v1 = lambda path, query=None, timeout=60: [
        {"id": "volume-1", "size": 100, "costPerHr": "Infinity"}]
    check("volume inventory rejects malformed size, rate, and duplicate ids",
          raises(RunPodError, bad_volume.list_network_volumes)
          and raises(RunPodError, duplicate_volume.list_network_volumes)
          and raises(RunPodError, bad_volume_rate.list_network_volumes))

    query = {"podId": "pod-1", "startTime": "2030-01-02T03:04:05Z",
             "endTime": "2030-01-02T04:04:05Z", "bucketSize": "hour"}
    amounts = {"totalAmount": "1.23", "gpuAmount": "1.00",
               "cpuAmount": "0.03", "diskAmount": "0.20"}
    resolved_query = dict(
        query, startTime="2030-01-02T03:00:00Z",
        endTime="2030-01-02T05:00:00Z")
    record = dict(
        amounts, podId="pod-1",
        startTime="2030-01-02T03:00:00Z",
        endTime="2030-01-02T04:00:00Z")
    zero_record = dict(
        {key: "0" for key in amounts}, podId="pod-1",
        startTime="2030-01-02T04:00:00Z",
        endTime="2030-01-02T05:00:00Z")
    response = {
        "records": [record, zero_record],
        "metadata": {
            "query": resolved_query, "recordCount": 2,
            "uniquePodCount": 1, "totals": amounts},
    }
    billing = BillingRunPod(response)
    evidence = billing.billing_history(
        "pod-1", start_time=query["startTime"], end_time=query["endTime"])
    check("official billing records bind snapped query and exact bucket range",
          evidence["pod_id"] == "pod-1"
          and evidence["records"][0]["totalAmount"] == "1.23"
          and evidence["validated_bucket_ranges"] == [{
              "startTime": "2030-01-02T03:00:00Z",
              "endTime": "2030-01-02T04:00:00Z"}, {
              "startTime": "2030-01-02T04:00:00Z",
              "endTime": "2030-01-02T05:00:00Z"}]
          and billing.billing_query == ("/billing/pods", query))
    # RunPod's live two-hour aggregate differed from its exact bucket sum by
    # 6e-18 USD. Preserve every provider decimal, but do not turn harmless
    # aggregate rounding into an unreconcilable lease.
    rounded = json.loads(json.dumps(response))
    rounded["metadata"]["totals"]["diskAmount"] = "0.200000000000000006"
    rounded_evidence = BillingRunPod(rounded).billing_history(
        "pod-1", start_time=query["startTime"], end_time=query["endTime"])
    excess_rounding = json.loads(json.dumps(response))
    excess_rounding["metadata"]["totals"]["diskAmount"] = (
        "0.200000000000002")
    check("billing accepts at most one femtodollar of provider rounding",
          rounded_evidence["validated_record_sums"]["diskAmount"] == "0.20"
          and raises(
              RunPodError,
              lambda: BillingRunPod(excess_rounding).billing_history(
                  "pod-1", start_time=query["startTime"],
                  end_time=query["endTime"])))
    missing = BillingRunPod({
        "records": [], "metadata": {
            "query": resolved_query, "recordCount": 0,
            "uniquePodCount": 0, "totals": {
                key: "0" for key in amounts}}})
    check("missing or lagging billing row is unresolved, never zero",
          raises(RunPodError, lambda: missing.billing_history(
              "pod-1", start_time=query["startTime"], end_time=query["endTime"])))
    coverage_query = dict(query, endTime="2030-01-02T05:04:05Z")
    coverage = json.loads(json.dumps(response))
    coverage["records"].append(dict(
        zero_record, startTime="2030-01-02T05:00:00Z",
        endTime="2030-01-02T06:00:00Z"))
    coverage["metadata"]["query"] = dict(
        resolved_query, endTime="2030-01-02T06:00:00Z")
    coverage["metadata"]["recordCount"] = 3

    def omitted_bucket(index, recalculate_totals=True):
        candidate = json.loads(json.dumps(coverage))
        candidate["records"].pop(index)
        candidate["metadata"]["recordCount"] = len(candidate["records"])
        if recalculate_totals:
            candidate["metadata"]["totals"] = {
                key: format(sum(
                    (Decimal(str(row[key])) for row in candidate["records"]),
                    Decimal("0")), "f")
                for key in amounts}
        return candidate

    # A provider that omits a charged edge bucket cannot hide the loss by
    # recalculating its totals: the original totals still include the charge
    # and the totals-vs-records-sum check catches the mismatch. An interior
    # gap is caught by the contiguity check regardless.
    check("billing refuses a charged edge bucket omitted with original totals "
          "and an interior gap",
          raises(RunPodError, lambda: BillingRunPod(
              omitted_bucket(0, recalculate_totals=False)).billing_history(
                  "pod-1", start_time=coverage_query["startTime"],
                  end_time=coverage_query["endTime"]))
          and raises(RunPodError, lambda: BillingRunPod(
              omitted_bucket(1, recalculate_totals=False)).billing_history(
                  "pod-1", start_time=coverage_query["startTime"],
                  end_time=coverage_query["endTime"])))

    # RunPod omits zero-charge edge buckets and aggregates partial-hour
    # charges into the nearest charged bucket. A pod that lived 06:57-07:22
    # resolves to 06:00-08:00 but returns only the 07:00-08:00 bucket. The
    # last zero-charge edge bucket (05:00-06:00) may be omitted; the totals
    # still match the record sums because the omitted bucket carried zero.
    last_omitted = omitted_bucket(2, recalculate_totals=False)
    last_omitted_evidence = BillingRunPod(last_omitted).billing_history(
        "pod-1", start_time=coverage_query["startTime"],
        end_time=coverage_query["endTime"])

    # Symmetrically, a zero-charge first bucket may be omitted. Build a
    # response where only the middle bucket carries charges, the first is
    # zero and omitted, and the totals reflect only the returned buckets.
    edge_gap = {
        "records": [dict(
            amounts, podId="pod-1",
            startTime="2030-01-02T04:00:00Z",
            endTime="2030-01-02T05:00:00Z")],
        "metadata": {
            "query": dict(
                resolved_query, endTime="2030-01-02T06:00:00Z"),
            "recordCount": 1, "uniquePodCount": 1, "totals": amounts}}
    edge_gap_evidence = BillingRunPod(edge_gap).billing_history(
        "pod-1", start_time=coverage_query["startTime"],
        end_time=coverage_query["endTime"])

    check("zero-charge edge buckets may be omitted when totals still match",
          last_omitted_evidence["validated_bucket_ranges"] == [{
              "startTime": "2030-01-02T03:00:00Z",
              "endTime": "2030-01-02T04:00:00Z"}, {
              "startTime": "2030-01-02T04:00:00Z",
              "endTime": "2030-01-02T05:00:00Z"}]
          and edge_gap_evidence["validated_bucket_ranges"] == [{
              "startTime": "2030-01-02T04:00:00Z",
              "endTime": "2030-01-02T05:00:00Z"}])


    multi = MultiBillingRunPod()
    mismatched = json.loads(json.dumps(response))
    mismatched["metadata"]["totals"]["totalAmount"] = "9.99"
    overlapping = json.loads(json.dumps(response))
    overlapping["records"].append(dict(
        record, startTime="2030-01-02T03:00:00Z",
        endTime="2030-01-02T04:00:00Z"))
    overlapping["metadata"]["recordCount"] = 3
    overlapping["metadata"]["totals"] = {
        key: str(Decimal(value) * 2) for key, value in amounts.items()}
    check("billing totals, record counts, overlap, and old data shape refuse",
          raises(RunPodError, lambda: BillingRunPod(mismatched).billing_history(
              "pod-1", start_time=query["startTime"],
              end_time=query["endTime"]))
          and raises(RunPodError, lambda:
                     BillingRunPod(overlapping).billing_history(
                         "pod-1", start_time=query["startTime"],
                         end_time=query["endTime"]))
          and raises(RunPodError, lambda:
                     BillingRunPod({"data": [record], "metadata":
                                    response["metadata"]}).billing_history(
                         "pod-1", start_time=query["startTime"],
                         end_time=query["endTime"])))
    # The absence lands at :44 of an hour; queried at :49 (m-dy325b,
    # 2026-09-05) RunPod had published only the previous hour's bucket, and
    # the closure sealed reconciled: true over half the bill. The hour that
    # contains the absence must close, plus the 300 s stabilization, before
    # a closure is returned; the reaper's later sweep settles it.
    reconciled_end_epoch = (int(time.time()) // 3600 - 2) * 3600 + 44 * 60
    reconciled_start_epoch = reconciled_end_epoch - 3600
    open_hour_lease = {
        "provider_resource_ids": ["pod-b", "pod-a"],
        "create": {"pre_create_observed_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(reconciled_start_epoch))},
        "history": [{"to": ABSENCE_CONFIRMED, "at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(reconciled_end_epoch))}],
    }
    check("billing closure refuses while the absence hour is open (queried at :49)",
          raises(RunPodError, lambda: multi.reconcile_billing(
              open_hour_lease, now=reconciled_end_epoch + 300))
          and raises(RunPodError, lambda: multi.reconcile_billing(
              open_hour_lease, now=reconciled_end_epoch + 16 * 60 + 200))
          and multi.asked == [])
    reconciled = multi.reconcile_billing(
        open_hour_lease, now=reconciled_end_epoch + 16 * 60 + 300)
    check("billing binds two independent identical reads for every exact id",
          multi.asked == ["pod-a", "pod-b", "pod-a", "pod-b"]
          and reconciled["provider_resource_ids"] == ["pod-a", "pod-b"]
          and reconciled["total_amount"] == "3.30"
          and len(reconciled["billing_histories"]) == 2
          and reconciled["evidence"]["schema"]
          == "fidelity-suite/runpod-billing-stabilization.v1"
          and reconciled["evidence"]["first_retrieval"]["retrieval_id"]
          != reconciled["evidence"]["second_retrieval"]["retrieval_id"])


def spawn_reaped_stage():
    helper = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess,sys; "
         "p=subprocess.Popen(['setsid','sleep','60']); "
         "print(p.pid,flush=True); raise SystemExit(p.wait())"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return helper, int(helper.stdout.readline().strip())


def watchdog_case():
    print("\n== watchdog unrelated-process safety ==")
    script = ROOT / "bin" / "watchdog.sh"
    with tempfile.TemporaryDirectory() as td:
        fs = Path(td)
        unrelated = subprocess.Popen(
            ["setsid", "sleep", "60"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            (fs / "receipts").mkdir()
            victim = fs / "atomic-victim"
            victim.write_text("unchanged\n", encoding="utf-8")
            receipt = fs / "receipts" / "watchdog-stage-pgid.json"
            os.symlink(str(victim), str(receipt))
            record = fs / "runtime" / "stage.pgid"
            armed = subprocess.run(
                ["bash", str(script), "--record-stage-pgid",
                 str(fs), str(unrelated.pid), str(record)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=10)
            lines = record.read_text(encoding="utf-8").splitlines()
            changed = []
            for line in lines:
                if line.startswith("start_ticks="):
                    changed.append("start_ticks=%d" % (int(line.split("=", 1)[1]) + 1))
                else:
                    changed.append(line)
            record.write_text("\n".join(changed) + "\n", encoding="utf-8")
            refused = subprocess.run(
                ["bash", str(script), str(int(time.time()) - 1), "60",
                 str(fs), str(record)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=10)
            abandoned = json.loads((fs / "ABANDONED.json").read_text(encoding="utf-8"))
            check("stale/reused PGID proof refuses signalling and leaves process alive",
                  armed.returncode == 0 and refused.returncode == 91
                  and unrelated.poll() is None
                  and victim.read_text(encoding="utf-8") == "unchanged\n"
                  and not receipt.is_symlink()
                  and abandoned["stage_process_group_stopped"] is False,
                  "arm=%s refuse=%s stderr=%s"
                  % (armed.returncode, refused.returncode, refused.stderr))
        finally:
            if unrelated.poll() is None:
                os.killpg(unrelated.pid, signal.SIGTERM)
            unrelated.wait(timeout=10)

    with tempfile.TemporaryDirectory() as td:
        fs = Path(td)
        helper, stage_pid = spawn_reaped_stage()
        try:
            record = fs / "runtime" / "stage.pgid"
            armed = subprocess.run(
                ["bash", str(script), "--record-stage-pgid",
                 str(fs), str(stage_pid), str(record)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=10)
            heartbeat = fs / "heartbeat"
            heartbeat.touch()
            future = int(time.time()) + 3600
            os.utime(str(heartbeat), (future, future))
            stopped = subprocess.run(
                ["bash", str(script), str(int(time.time()) + 600), "60",
                 str(fs), str(record)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=15)
            abandoned = json.loads(
                (fs / "ABANDONED.json").read_text(encoding="utf-8"))
            check("future heartbeat metadata abandons and stops exact group",
                  armed.returncode == 0 and stopped.returncode == 0
                  and "future" in abandoned["reason"]
                  and abandoned["stage_process_group_stopped"] is True
                  and not (fs / "logs" / "watchdog-seal.log").exists(),
                  stopped.stderr)
        finally:
            try:
                os.killpg(stage_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            helper.wait(timeout=10)

    with tempfile.TemporaryDirectory() as td:
        fs = Path(td)
        child_file = fs / "child.pid"
        leader = subprocess.Popen(
            ["setsid", "sh", "-c",
             "sleep 60 & echo $! > %s; sleep 1" % str(child_file)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            record = fs / "runtime" / "stage.pgid"
            armed = subprocess.run(
                ["bash", str(script), "--record-stage-pgid",
                 str(fs), str(leader.pid), str(record)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=10)
            leader.wait(timeout=5)
            stopped = subprocess.run(
                ["bash", str(script), str(int(time.time()) - 1), "60",
                 str(fs), str(record)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=15)
            abandoned = json.loads(
                (fs / "ABANDONED.json").read_text(encoding="utf-8"))
            check("leader exit does not hide surviving children in recorded PGID",
                  armed.returncode == 0 and stopped.returncode == 0
                  and abandoned["stage_process_group_stopped"] is True,
                  stopped.stderr)
        finally:
            try:
                os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if leader.poll() is None:
                leader.wait(timeout=10)


def stage_pgid_race_case():
    print("\n== stage pgid self-record race ==")
    import measure_cloud as mc
    watchdog = ROOT / "bin" / "watchdog.sh"
    image_ref = "test@sha256:" + "a" * 64
    image_digest = "sha256:" + "a" * 64

    def wrapper_command(fs_dir, stage_name, secrets_dir):
        return mc._runpod_stage_command(
            str(fs_dir), "/tmp/nonexistent-engine", stage_name,
            image_digest, image_ref, str(secrets_dir))

    # (a) A stage that exits 0 immediately after self-recording must return 0
    #     and leave the record + receipt -- not a spurious exit 70.
    with tempfile.TemporaryDirectory() as td:
        fs = Path(td)
        (fs / "bin").mkdir(parents=True)
        (fs / "logs").mkdir()
        shutil.copy(str(watchdog), str(fs / "bin" / "watchdog.sh"))
        stub = fs / "bin" / "stage_measure.sh"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'FS="$(readlink -f -- "$(dirname "$0")/..")"\n'
            'bash "$FS/bin/watchdog.sh" --record-stage-pgid "$FS" "$$"\n'
            "exit 0\n", encoding="utf-8")
        stub.chmod(0o755)
        secrets = fs / ".secrets"
        secrets.mkdir()
        cmd = wrapper_command(fs, "setup", secrets)
        proc = subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True,
            env=dict(os.environ, STAGE_PGID_WAIT_SECS="5"), timeout=30)
        record = fs / "runtime" / "stage.pgid"
        receipt = fs / "receipts" / "watchdog-stage-pgid.json"
        check("fast exit-0 stage returns 0 (not spurious 70)",
              proc.returncode == 0
              and record.is_file() and not record.is_symlink()
              and receipt.is_file() and not receipt.is_symlink(),
              "rc=%d record=%s receipt=%s stderr=%s"
              % (proc.returncode, record.exists(), receipt.exists(),
                 proc.stderr[:300]))

    # (d) The record path is per-RUN, the record is per-STAGE: a leftover
    #     record from the previous stage must not satisfy this stage's wait.
    #     A leader that records nothing and exits 42 after a delay would, with
    #     the stale file present, have been waited on at once and then -- the
    #     stale record being "present" -- reported as a recorded success path
    #     with its own code; but the record left behind would be the OLD
    #     leader's pgid, so the watchdog would target a dead group. The wrapper
    #     clears the record before launching; the stale content must be gone
    #     and the exit code must be the leader's.
    with tempfile.TemporaryDirectory() as td:
        fs = Path(td)
        (fs / "bin").mkdir(parents=True)
        (fs / "logs").mkdir()
        (fs / "runtime").mkdir()
        shutil.copy(str(watchdog), str(fs / "bin" / "watchdog.sh"))
        stale = fs / "runtime" / "stage.pgid"
        stale.write_text("version=1\nleader_pid=1\npgid=1\nsession_id=1\n"
                         "start_ticks=1\nrecorded_at_epoch=1\n", encoding="utf-8")
        stub = fs / "bin" / "stage_measure.sh"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "sleep 1\n"
            "exit 42\n", encoding="utf-8")
        stub.chmod(0o755)
        secrets = fs / ".secrets"
        secrets.mkdir()
        cmd = wrapper_command(fs, "capture", secrets)
        proc = subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True,
            env=dict(os.environ, STAGE_PGID_WAIT_SECS="5"), timeout=30)
        leftover = stale.read_text(encoding="utf-8") if stale.exists() else ""
        check("a stale record from the previous stage is cleared before launch and "
              "never stands in for this stage's own",
              proc.returncode == 42 and "leader_pid=1\n" not in leftover,
              "rc=%d stale_present=%s content=%r" % (proc.returncode, stale.exists(), leftover[:60]))

    # (c) A stage that exits non-zero after self-recording must propagate that
    #     code -- not 70, not 0.
    with tempfile.TemporaryDirectory() as td:
        fs = Path(td)
        (fs / "bin").mkdir(parents=True)
        (fs / "logs").mkdir()
        shutil.copy(str(watchdog), str(fs / "bin" / "watchdog.sh"))
        stub = fs / "bin" / "stage_measure.sh"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'FS="$(readlink -f -- "$(dirname "$0")/..")"\n'
            'bash "$FS/bin/watchdog.sh" --record-stage-pgid "$FS" "$$"\n'
            "exit 42\n", encoding="utf-8")
        stub.chmod(0o755)
        secrets = fs / ".secrets"
        secrets.mkdir()
        cmd = wrapper_command(fs, "measure", secrets)
        proc = subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True,
            env=dict(os.environ, STAGE_PGID_WAIT_SECS="5"), timeout=30)
        check("fast non-zero exit propagates its own code (42, not 70)",
              proc.returncode == 42,
              "rc=%d stderr=%s" % (proc.returncode, proc.stderr[:300]))

    # (b) A live leader that never records (unrecordable) is TERMed and
    #     yields 70.
    with tempfile.TemporaryDirectory() as td:
        fs = Path(td)
        (fs / "bin").mkdir(parents=True)
        (fs / "logs").mkdir()
        shutil.copy(str(watchdog), str(fs / "bin" / "watchdog.sh"))
        stub = fs / "bin" / "stage_measure.sh"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'FS="$(readlink -f -- "$(dirname "$0")/..")"\n'
            "sleep 300\n", encoding="utf-8")
        stub.chmod(0o755)
        secrets = fs / ".secrets"
        secrets.mkdir()
        cmd = wrapper_command(fs, "setup", secrets)
        proc = subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True,
            env=dict(os.environ, STAGE_PGID_WAIT_SECS="2"), timeout=30)
        record = fs / "runtime" / "stage.pgid"
        check("live unrecordable leader is TERMed and yields 70",
              proc.returncode == 70 and not record.exists(),
              "rc=%d record=%s stderr=%s"
              % (proc.returncode, record.exists(), proc.stderr[:300]))


def stage_progress_case():
    """The console progress line: the engine's meter (percent, ETA, rate) is
    preferred over the terse per-layer JSON that follows it in the log, the
    1.5 TB du runs only while fetch_target writes the tree, and a fetch line
    carries percent and a byte-rate ETA when the job binds model_bytes."""
    import measure_cloud as MC

    class _Prov:
        def __init__(self):
            self.commands = []

        def exec(self, pod_id, command, timeout=60, check=False):
            self.commands.append(command)
            # Simulate the shell: the log tail holds a meter line followed by
            # the per-layer JSON; `grep '^progress: ' || cat` picks the meter.
            if "grep '^progress: '" in command and "capture.log" in command:
                return {"stdout": "progress: layer-outer forwards 1951/2028 96% "
                                  "[07:16<00:17, 4.47 it/s]\n\n"}
            if "fetch_target.log" in command:
                return {"stdout": "Fetching 295 files:  24%\n\n338200000000\n"}
            return {"stdout": "\n\n"}

    prov = _Prov()
    line, landed = MC._runpod_stage_progress(prov, "pod", "/fs", "capture", measure_disk=False)
    check("stage progress prefers the engine's meter line over the trailing per-layer JSON",
          line is not None and line.startswith("progress: layer-outer forwards 1951/2028 96%")
          and landed is None, repr((line, landed)))
    check("du over models/target is not run for a non-fetch stage",
          "du -sb" not in prov.commands[-1], prov.commands[-1][-80:])
    line2, landed2 = MC._runpod_stage_progress(prov, "pod", "/fs", "fetch_target", measure_disk=True)
    check("fetch_target progress reads bytes landed and runs du",
          landed2 == 338200000000 and "du -sb" in prov.commands[-1], repr((line2, landed2)))
    text = MC._fetch_progress_text(338.2e9, 1506667387408, 1046e6, 300)
    check("a fetch line carries percent of the bound byte total and a byte-rate ETA",
          text == "338.2/1506.7 GB 22% (1046 MB/s, ~18m37s left)", text)
    check("without a bound total the line degrades to bytes and rate only",
          MC._fetch_progress_text(338.2e9, None, 1046e6, 300) == "338.2 GB on disk (1046 MB/s)",
          MC._fetch_progress_text(338.2e9, None, 1046e6, 300))


def main():
    lease_core_cases()
    reaper_cases()
    runpod_cases()
    watchdog_case()
    stage_pgid_race_case()
    stage_progress_case()
    print()
    if FAILED:
        print("selftest_reaper: %d FAILED" % len(FAILED))
        return 1
    print("selftest_reaper: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
