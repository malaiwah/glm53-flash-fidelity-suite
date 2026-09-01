#!/usr/bin/env python3
"""Focused offline checks for campaign-ledger-v1 admission accounting.

No provider or network access.  The concurrency case uses two independent
processes because the production exclusion mechanism is ``fcntl.flock``.
"""

import base64
import hashlib
import json
from decimal import Decimal
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))
import measure_cloud as measure_cloud  # noqa: E402

from fidelity.campaign import (  # noqa: E402
    CampaignLedger, CampaignLedgerError, CostQuote, RUNPOD_TARIFF_SOURCE,
)
from fidelity.cloudlease import (  # noqa: E402
    ABSENCE_CONFIRMED, LeaseError, LeaseStore,
    campaign_cleanup_binding_evidence, finalize_campaign_after_absence,
    reap_once,
)


FAILED = []
NOW = "2026-09-01T12:00:00+00:00"
FRESH_UNTIL = "2026-09-01T12:05:00+00:00"
PROVIDER = "runpod"
PROVIDER_ACCOUNT_ID = "runpod-account-selftest"
JOB_A = "a" * 64
JOB_B = "b" * 64
ATTEMPT_A = "1" * 24
ATTEMPT_B = "2" * 24
FIXTURE_PROVIDER_IDS = (
    "pod-123", "pod-rate-up", "pod-drift", "pod-lower", "pod-release",
    "pod-sequential", "pod-phases", "pod-deleted-phase", "pod-after-crash",
    "pod-crash-z", "pod:unclassified-7", "pod-drill",
)


def check(label, ok, detail=""):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)
        for line in str(detail).splitlines()[:8]:
            print("        %s" % line)


def quote(*, cap="6", reserved_rate="1", live_rate=None,
          container_size="20", container_rate="0.10",
          pod_size="20", pod_rate="0.10", network_size="0",
          network_rate="0.07", quoted_at=NOW, valid_until=FRESH_UNTIL,
          profile="runpod-ssh-safe",
          timing_kind="named-conservative-bound",
          timing_evidence="runpod-k6-conservative-bound-v1",
          workload="600", termination="900", retrieval="120", lag="60"):
    if live_rate is None:
        live_rate = reserved_rate
    return CostQuote(
        reserved_compute_usd_per_hour=Decimal(reserved_rate),
        live_compute_usd_per_hour=Decimal(live_rate),
        container_disk_size_gb=(None if container_size is None
                                else Decimal(container_size)),
        container_disk_running_usd_per_gb_month=(
            None if container_rate is None else Decimal(container_rate)),
        container_disk_stopped_usd_per_gb_month=(
            None if container_rate is None else Decimal("0")),
        pod_disk_size_gb=(None if pod_size is None else Decimal(pod_size)),
        pod_disk_running_usd_per_gb_month=(
            None if pod_rate is None else Decimal(pod_rate)),
        pod_disk_stopped_usd_per_gb_month=(
            None if pod_rate is None else Decimal("0.20")),
        network_volume_size_gb=(None if network_size is None
                                else Decimal(network_size)),
        network_volume_usd_per_gb_month=(None if network_rate is None
                                         else Decimal(network_rate)),
        storage_month_hours=Decimal("672"),  # conservative 28-day conversion
        network_billing_increment_seconds=Decimal("3600"),
        tariff_source=RUNPOD_TARIFF_SOURCE,
        tariff_effective_at="2026-09-01T00:00:00+00:00",
        quoted_at=quoted_at,
        valid_until=valid_until,
        target="GLM-5.3-Flash-TR3-K6",
        profile=profile,
        timing_kind=timing_kind,
        timing_evidence=timing_evidence,
        workload_deadline_seconds=Decimal(workload),
        provider_termination_deadline_seconds=Decimal(termination),
        retrieval_delete_reserve_seconds=Decimal(retrieval),
        timer_api_lag_seconds=Decimal(lag),
        hard_cap_usd=Decimal(cap),
    )


def ready_ledger(path, *, ceiling="10", floor="1", margin="1", width=2,
                 balance=None, balance_observed_at=NOW,
                 balance_valid_until=FRESH_UNTIL,
                 inventory_observed_at=NOW,
                 inventory_valid_until=FRESH_UNTIL,
                 inventory_complete=True, provider_resources=None):
    if balance is None:
        balance = ceiling
    if provider_resources is None:
        provider_resources = []
    ledger = CampaignLedger.create(
        str(path), Decimal(ceiling), Decimal(floor), Decimal(margin),
        max_concurrent_attempts=width, provider=PROVIDER,
        provider_account_id=PROVIDER_ACCOUNT_ID)
    result = ledger.record_provider_snapshot(
        0,
        provider=PROVIDER,
        provider_account_id=PROVIDER_ACCOUNT_ID,
        balance_available_usd=Decimal(balance),
        balance_observed_at=balance_observed_at,
        balance_valid_until=balance_valid_until,
        balance_source="runpod-balance-response-sha256:test",
        inventory_observed_at=inventory_observed_at,
        inventory_valid_until=inventory_valid_until,
        inventory_complete=inventory_complete,
        provider_resources=provider_resources,
        inventory_source="runpod-resource-list-sha256:test",
    )
    if not result.applied:
        raise RuntimeError(result)
    return ledger

def observe_pods(ledger, generation, *provider_ids):
    current = ledger.snapshot()
    balance = current["balance"]
    return ledger.record_provider_snapshot(
        generation,
        provider=PROVIDER,
        provider_account_id=PROVIDER_ACCOUNT_ID,
        balance_available_usd=Decimal(balance["available_usd"]),
        balance_observed_at=NOW,
        balance_valid_until=FRESH_UNTIL,
        balance_source="runpod-balance-response-sha256:observed-pods",
        inventory_observed_at=NOW,
        inventory_valid_until=FRESH_UNTIL,
        inventory_complete=True,
        provider_resources=[
            {"family": "pods", "id": provider_id,
             "name": "campaign-" + provider_id, "status": "RUNNING"}
            for provider_id in provider_ids
        ],
        inventory_source="runpod-resource-list-sha256:observed-pods")


def bind_observed_quote(ledger, generation, attempt_key, provider_id, actual_quote):
    observed = observe_pods(ledger, generation, provider_id)
    return ledger.bind_actual_quote(
        observed.generation, attempt_key, provider_id, actual_quote)


def bind_observed_cleanup(ledger, generation, attempt_key, provider_ids, evidence):
    observed = observe_pods(ledger, generation, *provider_ids)
    return ledger.bind_provider_for_cleanup(
        observed.generation, attempt_key, provider_ids, evidence)

def terminal_provider_snapshot(provider_resources=()):
    return {
        "provider": PROVIDER,
        "provider_account_id": PROVIDER_ACCOUNT_ID,
        "balance_available_usd": "100",
        "balance_observed_at": NOW,
        "balance_valid_until": FRESH_UNTIL,
        "balance_source": "runpod-balance-response-sha256:terminal",
        "inventory_observed_at": NOW,
        "inventory_valid_until": FRESH_UNTIL,
        "inventory_complete": True,
        "provider_resources": list(provider_resources),
        "inventory_source": "runpod-resource-list-sha256:terminal",
    }


def _concurrent_reserve(
        path, job_hash, attempt, start, output, effective_width=1):
    ledger = CampaignLedger(path, PROVIDER, PROVIDER_ACCOUNT_ID)
    start.wait()
    while True:
        generation = ledger.snapshot()["generation"]
        result = ledger.reserve(
            generation, job_hash, attempt, quote(), NOW,
            effective_width=effective_width)
        if result.code != "GENERATION_CONFLICT":
            output.put(result.to_dict())
            return


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # C0: the paid controller-loss drill is the first reservation in the
        # same cumulative ledger.  Its exact deletion and billing settle
        # normally, and the later measurement sees its lifetime charge.
        bootstrap = ready_ledger(
            root / "bootstrap-drill.json", ceiling="100", width=2,
            provider_resources=[])
        invalid_bootstrap_quote_refused = False
        try:
            bootstrap.reserve_bootstrap_drill(
                1, "c" * 64, "3" * 24, quote(cap="3"), NOW)
        except ValueError:
            invalid_bootstrap_quote_refused = True
        drill_quote = quote(
            cap="3", profile="runpod-drill-secure-l4-on-demand",
            timing_kind="exact-target-profile", timing_evidence="9" * 64)
        drill_reserved = bootstrap.reserve_bootstrap_drill(
            1, "c" * 64, "3" * 24, drill_quote, NOW)
        drill_creating = bootstrap.mark_creating(
            drill_reserved.generation, drill_reserved.attempt_key)
        drill_inventory = bootstrap.record_provider_snapshot(
            drill_creating.generation,
            provider=PROVIDER,
            provider_account_id=PROVIDER_ACCOUNT_ID,
            balance_available_usd=Decimal("100"),
            balance_observed_at=NOW,
            balance_valid_until=FRESH_UNTIL,
            balance_source="runpod-balance-response-sha256:drill",
            inventory_observed_at=NOW,
            inventory_valid_until=FRESH_UNTIL,
            inventory_complete=True,
            provider_resources=[
                {"family": "pods", "id": "pod-drill",
                 "name": "fidelity-drill", "status": "RUNNING"},
            ],
            inventory_source="runpod-resource-list-sha256:drill")
        drill_bound = bootstrap.bind_actual_quote(
            drill_inventory.generation, drill_reserved.attempt_key,
            "pod-drill", drill_quote)
        drill_deleted = bootstrap.mark_deleted(
            drill_bound.generation, drill_reserved.attempt_key, "pod-drill",
            NOW, "drill-delete-and-absence-sha256:test")
        post_drill_inventory = bootstrap.record_provider_snapshot(
            drill_deleted.generation,
            provider=PROVIDER,
            provider_account_id=PROVIDER_ACCOUNT_ID,
            balance_available_usd=Decimal("98.75"),
            balance_observed_at=NOW,
            balance_valid_until=FRESH_UNTIL,
            balance_source="runpod-balance-response-sha256:post-drill",
            inventory_observed_at=NOW,
            inventory_valid_until=FRESH_UNTIL,
            inventory_complete=True,
            provider_resources=[],
            inventory_source="runpod-resource-list-sha256:post-drill")
        drill_billed = bootstrap.reconcile_billing(
            post_drill_inventory.generation, drill_reserved.attempt_key,
            "pod-drill",
            NOW, Decimal("1.25"), "drill-billing-line-sha256:test")
        measurement_after_drill = bootstrap.reserve(
            drill_billed.generation, JOB_A, ATTEMPT_A, quote(cap="2"), NOW)
        second_bootstrap = bootstrap.reserve_bootstrap_drill(
            measurement_after_drill.generation, "d" * 64, "4" * 24,
            drill_quote, NOW)
        bootstrap_doc = bootstrap.snapshot()
        check("C0 paid bootstrap drill remains in cumulative campaign accounting",
              invalid_bootstrap_quote_refused
              and drill_reserved.admitted
              and bootstrap_doc["attempts"][drill_reserved.attempt_key][
                  "reservation_kind"] == "bootstrap-controller-loss-drill"
              and bootstrap_doc["settled_charges_usd"] == "1.25"
              and measurement_after_drill.admitted
              and second_bootstrap.code == "BOOTSTRAP_DRILL_NOT_FIRST"
              and len(bootstrap_doc["attempts"]) == 2,
              (invalid_bootstrap_quote_refused, drill_reserved, drill_billed,
               measurement_after_drill, second_bootstrap, bootstrap_doc))

        # C0b: a later full-family response can disprove an earlier absence.
        # That response must freeze before any deletion proof reaches campaign
        # state; otherwise the later true absence can never replace the stale
        # proof and campaign liability remains stranded.
        reappeared_path = root / "reappeared-campaign.json"
        reappeared = ready_ledger(
            reappeared_path, ceiling="100", width=2,
            provider_resources=[])
        reappeared_quote = quote(
            cap="3", profile="runpod-drill-secure-l4-on-demand",
            timing_kind="exact-target-profile", timing_evidence="8" * 64)
        reappeared_reserved = reappeared.reserve_bootstrap_drill(
            1, "e" * 64, "5" * 24, reappeared_quote, NOW)
        reappeared_creating = reappeared.mark_creating(
            reappeared_reserved.generation, reappeared_reserved.attempt_key)
        reappeared_inventory = observe_pods(
            reappeared, reappeared_creating.generation, "pod-reappeared")
        reappeared_bound = reappeared.bind_actual_quote(
            reappeared_inventory.generation,
            reappeared_reserved.attempt_key,
            "pod-reappeared", reappeared_quote)

        class ReappearedProvider:
            @staticmethod
            def chargeable_inventory():
                family = lambda rows: {
                    "complete": True, "resources": rows, "source": "fixture"}
                return {
                    "schema": "fidelity-suite/runpod-chargeable-inventory.v1",
                    "provider": PROVIDER, "complete": True,
                    "unknown_families": [],
                    "observed_at_utc": "2026-09-01T12:00:01Z",
                    "families": {
                        "pods": family([{
                            "id": "pod-reappeared",
                            "name": "campaign-pod-reappeared",
                            "status": "RUNNING",
                        }]),
                        "network_volumes": family([]),
                    },
                }
            @staticmethod
            def list_lifecycle_resources():
                return [{
                    "id": "pod-reappeared",
                    "name": "campaign-pod-reappeared",
                    "status": "RUNNING",
                }]


            @staticmethod
            def status():
                return {
                    "id": PROVIDER_ACCOUNT_ID,
                    "clientBalance": "100",
                    "observed_at_utc": "2026-09-01T12:00:02Z",
                }

        reappeared_document = {
            "job_hash": "e" * 64,
            "attempt_id": "5" * 24,
            "provider_resource_ids": ["pod-reappeared"],
            "create": {
                "provider": PROVIDER,
                "request_sha256": "7" * 64,
                "request": {
                    "provider_account_id": PROVIDER_ACCOUNT_ID,
                    "campaign_ledger": reappeared_path.name,
                    "campaign_attempt_key": reappeared_reserved.attempt_key,
                },
            },
            "terminal_proof": {
                "provider_absence": {
                    "target_provider_ids": ["pod-reappeared"],
                    "still_present_ids": [],
                },
            },
            "billing_reconciliation": {"total_amount": "0.01"},
            "history": [{
                "event": "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING",
                "at": "2026-09-01T12:00:00Z",
            }, {
                "event": "BILLING_RECONCILIATION_STAGED_FOR_CAMPAIGN",
                "at": "2026-09-01T12:00:03Z",
            }],
        }
        reappeared_refused = False
        try:
            finalize_campaign_after_absence(
                ReappearedProvider(), reappeared_document,
                (root / "reappeared-leases").resolve())
        except LeaseError as exc:
            reappeared_refused = "reappeared" in str(exc)
        reappeared_after = reappeared.snapshot()
        check("C0b fresh target reappearance cannot project false deletion",
              reappeared_refused
              and reappeared_after["generation"]
                  == reappeared_bound.generation
              and reappeared_after["attempts"][
                  reappeared_reserved.attempt_key]["phase"] == "LIVE"
              and reappeared_after["attempts"][
                  reappeared_reserved.attempt_key]["deletion"] is None,
              reappeared_after)

        class UnknownResourceProvider(ReappearedProvider):
            @staticmethod
            def chargeable_inventory():
                family = lambda rows: {
                    "complete": True, "resources": rows, "source": "fixture"}
                return {
                    "schema": "fidelity-suite/runpod-chargeable-inventory.v1",
                    "provider": PROVIDER, "complete": True,
                    "unknown_families": [],
                    "observed_at_utc": "2026-09-01T12:00:01Z",
                    "families": {
                        "pods": family([]),
                        "network_volumes": family([{
                            "id": "foreign-volume",
                            "name": "unowned-volume",
                            "status": "READY",
                        }]),
                    },
                }
            @staticmethod
            def list_lifecycle_resources():
                return []


        unknown_refused = False
        try:
            finalize_campaign_after_absence(
                UnknownResourceProvider(), reappeared_document,
                (root / "reappeared-leases").resolve())
        except LeaseError as exc:
            unknown_refused = "unknown chargeable resources" in str(exc)
        unknown_after = reappeared.snapshot()
        check("C0c unknown fresh resources cannot release campaign liability",
              unknown_refused
              and unknown_after["generation"] == reappeared_bound.generation
              and unknown_after["attempts"][
                  reappeared_reserved.attempt_key]["phase"] == "LIVE"
              and unknown_after["attempts"][
                  reappeared_reserved.attempt_key]["released"] is False,
              unknown_after)
        class GraphqlOnlyReappearedProvider(UnknownResourceProvider):
            @staticmethod
            def chargeable_inventory():
                family = lambda rows: {
                    "complete": True, "resources": rows, "source": "fixture"}
                return {
                    "schema": "fidelity-suite/runpod-chargeable-inventory.v1",
                    "provider": PROVIDER, "complete": True,
                    "unknown_families": [],
                    "observed_at_utc": "2026-09-01T12:00:04Z",
                    "families": {
                        "pods": family([]),
                        "network_volumes": family([]),
                    },
                }

            @staticmethod
            def list_lifecycle_resources():
                return [{
                    "id": "pod-reappeared",
                    "name": "campaign-pod-reappeared",
                    "status": "RUNNING",
                }]

        graphql_refused = False
        try:
            finalize_campaign_after_absence(
                GraphqlOnlyReappearedProvider(), reappeared_document,
                (root / "reappeared-leases").resolve())
        except LeaseError as exc:
            graphql_refused = "reappeared" in str(exc)
        graphql_after = reappeared.snapshot()
        check("C0d GraphQL-only target cannot project false deletion",
              graphql_refused
              and graphql_after["generation"] == reappeared_bound.generation
              and graphql_after["attempts"][
                  reappeared_reserved.attempt_key]["phase"] == "LIVE"
              and graphql_after["attempts"][
                  reappeared_reserved.attempt_key]["released"] is False,

              graphql_after)
        repair_path = root / "repair-campaign.json"
        repair = ready_ledger(
            repair_path, ceiling="100", width=2, provider_resources=[])
        repair_reserved = repair.reserve(
            1, "9" * 64, "8" * 24, quote(cap="3"), NOW)
        repair_creating = repair.mark_creating(
            repair_reserved.generation, repair_reserved.attempt_key)
        repair_store = LeaseStore(
            root / "repair-leases", clock=lambda: 1000.0)
        repair_job = "9" * 64
        repair_attempt = "8" * 24
        repair_name = "fidcloud-%s-a%s" % (
            repair_job, repair_attempt)
        repair_terminate_after = "2030-01-02T03:04:05Z"
        repair_body = json.dumps({
            "query": "mutation { podFindAndDeployOnDemand }"
        }).encode("utf-8")
        repair_prepared = {
            "schema": "fidelity-suite/runpod-prepared-create.v1",
            "request_identity": {
                "cloud_type": "SECURE", "is_spot": False,
                "offer": "on-demand", "gpu_type_id": "A100",
                "gpu_count": 1, "volume_gb": 100,
                "container_disk_gb": 20, "min_vcpu": 4,
                "min_ram_gb": 16, "name": repair_name,
                "image_name": "image",
                "terminate_after": repair_terminate_after,
                "ports": "22/tcp", "volume_mount_path": "/workspace",
                "network_volume_id": None,
                "public_key_sha256": "0" * 64,
            },
            "graphql_body_sha256": hashlib.sha256(
                repair_body).hexdigest(),
            "graphql_body_bytes": len(repair_body),
            "graphql_body_base64": base64.b64encode(
                repair_body).decode("ascii"),
        }
        repair_request = {
            "attempt_key": repair_attempt,
            "campaign_attempt_key": repair_reserved.attempt_key,
            "campaign_ledger": repair_path.name,
            "provider": PROVIDER,
            "provider_account_id": PROVIDER_ACCOUNT_ID,
            "gpu_type": "A100", "normalized_gpu": "A100",
            "num_gpus": 1, "secure_cloud": True,
            "storage_gb": 100, "remote_root": "/workspace/f",
            "engine_root": "/workspace/e", "container_disk_gb": 20,
            "image": "image", "min_vcpu_count": 4,
            "min_memory_gb": 16, "workload_contract": {},
            "offer": "on-demand", "network_volume": None,
            "terminate_after": repair_terminate_after, "quote": {},
            "pre_create_safety": {
                "schema": "fidelity-suite/runpod-server-time.v1",
                "endpoint_origin": "https://api.runpod.io",
                "date_header": "Thu, 01 Jan 1970 00:16:40 GMT",
                "server_epoch": 1000.0, "local_received_epoch": 1000.0,
                "local_minus_server_seconds": 0.0,
                "checked_at_epoch": 1000.0, "evidence_age_seconds": 0.0,
                "max_clock_delta_seconds": 30.0,
                "max_evidence_age_seconds": 30.0,
            },
            "execution_contract_sha256": "d" * 64,
            "grounding_bundle": {
                "schema": "fidelity-suite/grounding-bundle.v1",
                "archive_sha256": "e" * 64, "archive_bytes": 1,
                "manifest_sha256": "f" * 64,
            },
            "prepared_create": repair_prepared,
        }
        repair_ref = repair_store.begin_create(
            job_hash=repair_job, provider=PROVIDER,
            request=repair_request,
            pre_create_resources=[], attempt_id=repair_attempt,
            controller_pid=2 ** 30,
            create_deadline_epoch=1100,
            workload_deadline_epoch=4600)
        repair_ref = repair_store.record_post_intent(repair_ref)
        repair_document = repair_store.read(repair_ref)
        repair_bound = repair.bind_provider_for_cleanup(
            repair_creating.generation, repair_reserved.attempt_key,
            ["campaign-only-id"],
            campaign_cleanup_binding_evidence(
                repair_document, ["campaign-only-id"]))

        class CampaignRepairProvider:
            def __init__(self):
                self.present = True
                self.destroyed = []

            @staticmethod
            def status():
                return {
                    "id": PROVIDER_ACCOUNT_ID,
                    "clientBalance": "100",
                    "observed_at_utc": "1970-01-01T00:16:40Z",
                }

            def list_instances(self):
                return ([{
                    "id": "campaign-only-id",
                    "name": "provider-rewrote-name",
                    "status": "RUNNING",
                }] if self.present else [])

            def chargeable_inventory(self):
                family = lambda rows: {
                    "complete": True, "resources": rows, "source": "fixture"}
                return {
                    "schema": "fidelity-suite/runpod-chargeable-inventory.v1",
                    "provider": PROVIDER, "complete": True,
                    "unknown_families": [],
                    "observed_at_utc": "1970-01-01T00:16:40Z",
                    "families": {
                        "pods": family(self.list_instances()),
                        "network_volumes": family([]),
                    },
                }

            def destroy(self, provider_id):
                self.destroyed.append(provider_id)
                self.present = False

        repair_provider = CampaignRepairProvider()
        repair_result = reap_once(
            repair_store, {PROVIDER: repair_provider}, now=2000.0)
        repaired_document = repair_store.read(repair_ref)
        repaired_ambiguity = (
            repaired_document["terminal_proof"]["ambiguous_create"])
        check("C0e reaper imports campaign-only committed response ID",
              repair_bound.applied
              and repair_provider.destroyed == ["campaign-only-id"]
              and repaired_document["state"] == ABSENCE_CONFIRMED
              and repaired_document["provider_resource_ids"]
                  == ["campaign-only-id"]
              and repaired_ambiguity[
                  "unattributable_wrong_name_pod_ids"] == []
              and any("billing" in failure["error"]
                      for failure in repair_result.failures),
              (repair_result, repaired_document))


        # C1: two separately locked controller processes cannot both consume a
        # ceiling that fits only one maximum liability.
        concurrent_path = root / "concurrent.json"
        concurrent_ledger = ready_ledger(concurrent_path)
        concurrent_ledger.authorize_concurrent_width_two(
            1, NOW, "c" * 64, "d" * 64)
        ctx = multiprocessing.get_context("fork")
        start = ctx.Event()
        output = ctx.Queue()
        processes = [
            ctx.Process(target=_concurrent_reserve,
                        args=(str(concurrent_path), JOB_A, ATTEMPT_A, start,
                              output, 2)),
            ctx.Process(target=_concurrent_reserve,
                        args=(str(concurrent_path), JOB_B, ATTEMPT_B, start,
                              output, 2)),
        ]
        for process in processes:
            process.start()
        start.set()
        results = [output.get(timeout=10), output.get(timeout=10)]
        for process in processes:
            process.join(timeout=10)
        final = CampaignLedger(
            str(concurrent_path), PROVIDER, PROVIDER_ACCOUNT_ID).snapshot()
        check("C1 independent controllers admit exactly one attempt",
              sum(result["admitted"] for result in results) == 1
              and sorted(result["code"] for result in results)
              == ["ADMITTED", "CEILING_EXCEEDED"]
              and len(final["attempts"]) == 1,
              (results, final))

        # C1b: width is enforced in the same locked reserve mutation, not by a
        # stale caller-side count.  Both processes begin against width one.
        width_path = root / "width.json"
        ready_ledger(width_path, ceiling="100", width=1)
        width_start = ctx.Event()
        width_output = ctx.Queue()
        width_processes = [
            ctx.Process(target=_concurrent_reserve,
                        args=(str(width_path), JOB_A, ATTEMPT_A,
                              width_start, width_output)),
            ctx.Process(target=_concurrent_reserve,
                        args=(str(width_path), JOB_B, ATTEMPT_B,
                              width_start, width_output)),
        ]
        for process in width_processes:
            process.start()
        width_start.set()
        width_results = [
            width_output.get(timeout=10), width_output.get(timeout=10)]
        for process in width_processes:
            process.join(timeout=10)
        width_identity_refused = False
        try:
            CampaignLedger.create(
                str(width_path), Decimal("100"), Decimal("1"), Decimal("1"),
                max_concurrent_attempts=2, provider=PROVIDER,
                provider_account_id=PROVIDER_ACCOUNT_ID)
        except CampaignLedgerError:
            width_identity_refused = True
        width_doc = CampaignLedger(
            str(width_path), PROVIDER, PROVIDER_ACCOUNT_ID).snapshot()
        check("C1b width-one reservation is atomic durable campaign identity",
              sorted(result["code"] for result in width_results)
              == ["ADMITTED", "WIDTH_EXCEEDED"]
              and len(width_doc["attempts"]) == 1
              and width_doc["max_concurrent_attempts"] == 1
              and width_identity_refused,
              (width_results, width_doc))

        # C1c: maximum width two is immutable from creation, but effective
        # width stays one until durable Fruit public evidence authorizes the
        # sole monotonic transition.  The ledger and all prior attempts remain.
        fruit_width = ready_ledger(
            root / "fruit-width.json", ceiling="100", width=2)
        fruit_first = fruit_width.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="2"), NOW)
        before_fruit_proof = fruit_width.reserve(
            fruit_first.generation, JOB_B, ATTEMPT_B, quote(cap="2"), NOW)
        fruit_authorized = fruit_width.authorize_concurrent_width_two(
            fruit_first.generation, NOW, "e" * 64, "f" * 64)
        effective_one_after_proof = fruit_width.reserve(
            fruit_authorized.generation, JOB_B, ATTEMPT_B,
            quote(cap="2"), NOW)
        after_fruit_proof = fruit_width.reserve(
            fruit_authorized.generation, JOB_B, ATTEMPT_B,
            quote(cap="2"), NOW, effective_width=2)
        fruit_authorized_retry = fruit_width.authorize_concurrent_width_two(
            after_fruit_proof.generation, NOW, "e" * 64, "f" * 64)
        fruit_authorized_mismatch = fruit_width.authorize_concurrent_width_two(
            after_fruit_proof.generation, NOW, "e" * 64, "0" * 64)
        fruit_reopened = CampaignLedger.create(
            str(root / "fruit-width.json"), Decimal("100"), Decimal("1"),
            Decimal("1"), max_concurrent_attempts=2, provider=PROVIDER,
            provider_account_id=PROVIDER_ACCOUNT_ID)
        fruit_width_doc = fruit_reopened.snapshot()
        check("C1c Fruit evidence atomically authorizes width one to two",
              before_fruit_proof.code == "WIDTH_EXCEEDED"
              and fruit_authorized.code == "WIDTH_TWO_AUTHORIZED"
              and effective_one_after_proof.code == "WIDTH_EXCEEDED"
              and after_fruit_proof.admitted
              and fruit_authorized_retry.code == "WIDTH_ALREADY_AUTHORIZED"
              and fruit_authorized_mismatch.code
                  == "WIDTH_AUTHORIZATION_MISMATCH"
              and fruit_width_doc["authorized_concurrent_attempts"] == 2
              and fruit_width_doc["width_authorization"][
                  "fruit_public_archive_sha256"] == "e" * 64
              and len(fruit_width_doc["attempts"]) == 2,
              (before_fruit_proof, fruit_authorized,
               effective_one_after_proof, after_fruit_proof,
               fruit_authorized_retry, fruit_authorized_mismatch,
               fruit_width_doc))

        # C2: CREATING is already chargeable.  Until its POST resolves to one
        # exact provider ID, it both stays in the sum and freezes admission.
        creating = ready_ledger(root / "creating.json", ceiling="30")
        first = creating.reserve(1, JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        marked = creating.mark_creating(first.generation, first.attempt_key)
        refused = creating.reserve(marked.generation, JOB_B, ATTEMPT_B,
                                   quote(cap="4"), NOW)
        creating_doc = creating.snapshot()
        check("C2 CREATING/null-ID liability is retained and freezes admission",
              marked.applied and refused.code == "AMBIGUOUS_CREATE"
              and creating_doc["attempts"][first.attempt_key]
                  ["maximum_remaining_liability_usd"] == "4",
              (marked, refused, creating_doc))

        # C3: EXITED and TERMINATE_REQUESTED are lifecycle states, not proof of
        # provider deletion; neither is allowed to reduce the reservation.
        lifecycle = ready_ledger(root / "lifecycle.json", ceiling="30")
        admitted = lifecycle.reserve(1, JOB_A, ATTEMPT_A, quote(cap="5"), NOW)
        creating_result = lifecycle.mark_creating(admitted.generation,
                                                  admitted.attempt_key)
        bound = bind_observed_quote(
            lifecycle, creating_result.generation,
            admitted.attempt_key, "pod-123", quote(cap="5"))
        exited = lifecycle.mark_phase(bound.generation, admitted.attempt_key, "EXITED")
        terminating = lifecycle.mark_phase(exited.generation, admitted.attempt_key,
                                            "TERMINATE_REQUESTED")
        lifecycle_item = lifecycle.snapshot()["attempts"][admitted.attempt_key]
        check("C3 EXITED/TERMINATE_REQUESTED retains full liability",
              terminating.applied
              and lifecycle_item["phase"] == "TERMINATE_REQUESTED"
              and lifecycle_item["maximum_remaining_liability_usd"] == "5",
              lifecycle_item)

        # C4: balance and inventory freshness are independently checked at the
        # admission mutation instant.
        stale = ready_ledger(
            root / "stale.json", ceiling="30",
            balance_observed_at="2026-09-01T11:55:00+00:00",
            balance_valid_until="2026-09-01T11:59:59+00:00")
        stale_result = stale.reserve(1, JOB_A, ATTEMPT_A, quote(cap="2"), NOW)
        stale_inventory = ready_ledger(
            root / "stale-inventory.json", ceiling="30",
            inventory_observed_at="2026-09-01T11:55:00+00:00",
            inventory_valid_until="2026-09-01T11:59:59+00:00")
        stale_inventory_result = stale_inventory.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="2"), NOW)
        check("C4 stale provider balance or inventory refuses admission",
              not stale_result.admitted and stale_result.code == "BALANCE_STALE"
              and not stale_inventory_result.admitted
              and stale_inventory_result.code == "INVENTORY_STALE",
              (stale_result, stale_inventory_result))

        future_balance = ready_ledger(
            root / "future-balance.json", ceiling="30",
            balance_observed_at="2026-09-01T12:01:00+00:00",
            balance_valid_until="2026-09-01T12:05:00+00:00")
        future_balance_result = future_balance.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="2"), NOW)
        future_inventory = ready_ledger(
            root / "future-inventory.json", ceiling="30",
            inventory_observed_at="2026-09-01T12:01:00+00:00",
            inventory_valid_until="2026-09-01T12:05:00+00:00")
        future_inventory_result = future_inventory.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="2"), NOW)
        future_quote = ready_ledger(
            root / "future-quote.json", ceiling="30")
        future_quote_result = future_quote.reserve(
            1, JOB_A, ATTEMPT_A,
            quote(cap="2", quoted_at="2026-09-01T12:01:00+00:00",
                  valid_until="2026-09-01T12:06:00+00:00"), NOW)
        boundary_time = ready_ledger(
            root / "freshness-boundary.json", ceiling="30",
            balance_valid_until=NOW, inventory_valid_until=NOW)
        boundary_time_result = boundary_time.reserve(
            1, JOB_A, ATTEMPT_A,
            quote(cap="2", quoted_at=NOW, valid_until=NOW), NOW)
        overlong_quote_refused = False
        try:
            quote(cap="2", valid_until="2026-09-01T12:05:01+00:00")
        except ValueError:
            overlong_quote_refused = True
        check("C4b freshness rejects future evidence, includes both boundaries, "
              "and caps quote lifetime",
              future_balance_result.code == "BALANCE_FUTURE"
              and future_inventory_result.code == "INVENTORY_FUTURE"
              and future_quote_result.code == "QUOTE_FUTURE"
              and boundary_time_result.admitted
              and overlong_quote_refused,
              (future_balance_result, future_inventory_result,
               future_quote_result, boundary_time_result,
               overlong_quote_refused))

        # C5: missing local-storage size/rate is not treated as zero.  Likewise,
        # a complete inventory containing an unclassified provider resource
        # freezes admission.
        unknown_storage = ready_ledger(root / "unknown-storage.json", ceiling="30")
        storage_result = unknown_storage.reserve(
            1, JOB_A, ATTEMPT_A,
            quote(cap="2", container_size=None), NOW)
        unknown_resource = ready_ledger(
            root / "unknown-resource.json", ceiling="30",
            provider_resources=[
                {"family": "pods", "id": "pod:unclassified-7",
                 "name": "unledgered", "status": "RUNNING"},
            ])
        resource_result = unknown_resource.reserve(1, JOB_A, ATTEMPT_A,
                                                   quote(cap="2"), NOW)
        check("C5 unknown storage and resources fail closed",
              storage_result.code == "UNKNOWN_STORAGE"
              and resource_result.code == "UNKNOWN_RESOURCES",
              (storage_result, resource_result))

        resource_map = [
            {"family": "network_volumes", "id": "shared-resource",
             "name": "volume-name", "status": "AVAILABLE"},
            {"family": "pods", "id": "shared-resource",
             "name": "pod-name", "status": "RUNNING"},
        ]
        mapped_inventory = ready_ledger(
            root / "mapped-inventory.json", ceiling="30",
            provider_resources=resource_map)
        mapped_doc = mapped_inventory.snapshot()
        empty_inventory = ready_ledger(
            root / "empty-inventory.json", ceiling="30",
            provider_resources=[])
        empty_admission = empty_inventory.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="2"), NOW)
        classified_map = mapped_inventory.classify_provider_resources(
            resource_map)
        missing_active = ready_ledger(
            root / "missing-active-map.json", ceiling="30")
        missing_reserved = missing_active.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="2"), NOW)
        missing_creating = missing_active.mark_creating(
            missing_reserved.generation, missing_reserved.attempt_key)
        volume_observed = missing_active.record_provider_snapshot(
            missing_creating.generation,
            provider=PROVIDER,
            provider_account_id=PROVIDER_ACCOUNT_ID,
            balance_available_usd=Decimal("30"),
            balance_observed_at=NOW,
            balance_valid_until=FRESH_UNTIL,
            balance_source="runpod-balance-response-sha256:family-collision",
            inventory_observed_at=NOW,
            inventory_valid_until=FRESH_UNTIL,
            inventory_complete=True,
            provider_resources=[
                {"family": "network_volumes", "id": "pod-not-observed",
                 "name": "same-id-volume", "status": "AVAILABLE"},
            ],
            inventory_source="runpod-resource-list-sha256:family-collision")
        missing_active_refused = False
        try:
            missing_active.bind_actual_quote(
                volume_observed.generation, missing_reserved.attempt_key,
                "pod-not-observed", quote(cap="2"))
        except ValueError:
            missing_active_refused = True
        check("C5b inventory persists exact sorted resources and binds active IDs",
              [(item["family"], item["id"]) for item in
               mapped_doc["inventory"]["provider_resources"]]
                  == [("network_volumes", "shared-resource"),
                      ("pods", "shared-resource")]
              and classified_map["unknown_resources"] == [
                  {"family": "network_volumes", "id": "shared-resource"},
                  {"family": "pods", "id": "shared-resource"},
              ]
              and missing_active_refused,
              (mapped_doc["inventory"], empty_admission,
               classified_map, missing_active_refused))

        network_resource_result = mapped_inventory.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="2"), NOW)
        sibling = ready_ledger(
            root / "known-sibling.json", ceiling="100", width=2)
        sibling.authorize_concurrent_width_two(
            1, NOW, "7" * 64, "8" * 64)
        sibling_first = sibling.reserve(
            2, JOB_A, ATTEMPT_A, quote(cap="2"), NOW)
        sibling_creating = sibling.mark_creating(
            sibling_first.generation, sibling_first.attempt_key)
        sibling_bound = bind_observed_quote(
            sibling, sibling_creating.generation, sibling_first.attempt_key,
            "pod-known-sibling", quote(cap="2"))
        sibling_classified = sibling.classify_provider_resources([
            {"family": "pods", "id": "pod-known-sibling",
             "name": "campaign-sibling", "status": "RUNNING"},
        ])
        sibling_second = sibling.reserve(
            sibling_bound.generation, JOB_B, ATTEMPT_B, quote(cap="2"), NOW,
            effective_width=2)

        race = ready_ledger(root / "unbound-create-race.json", ceiling="100")
        race.authorize_concurrent_width_two(
            1, NOW, "5" * 64, "6" * 64)
        race_first = race.reserve(
            2, JOB_A, ATTEMPT_A, quote(cap="2"), NOW)
        race_creating = race.mark_creating(
            race_first.generation, race_first.attempt_key)
        race_inventory = race.record_provider_snapshot(
            race_creating.generation,
            provider=PROVIDER,
            provider_account_id=PROVIDER_ACCOUNT_ID,
            balance_available_usd=Decimal("100"),
            balance_observed_at=NOW,
            balance_valid_until=FRESH_UNTIL,
            balance_source="runpod-balance-response-sha256:create-race",
            inventory_observed_at=NOW,
            inventory_valid_until=FRESH_UNTIL,
            inventory_complete=True,
            provider_resources=[
                {"family": "pods", "id": "pod-unbound-race",
                 "name": "exact-attempt-name", "status": "RUNNING"},
            ],
            inventory_source="runpod-resource-list-sha256:create-race")
        race_second = race.reserve(
            race_inventory.generation, JOB_B, ATTEMPT_B, quote(cap="2"), NOW)
        check("C5c canonical classifier refuses foreign resources and permits "
              "only durably bound sibling pods",
              network_resource_result.code == "UNKNOWN_RESOURCES"
              and sibling_classified["unknown_resources"] == []
              and sibling_second.admitted
              and race_second.code == "UNKNOWN_RESOURCES",
              (network_resource_result, sibling_classified, sibling_second,
               race_inventory, race_second))

        # C6: exact Decimal equality is admissible; a binary float conversion
        # must never perturb the boundary.
        boundary = ready_ledger(root / "decimal.json", ceiling="0.3",
                                floor="0.1", margin="0.1")
        boundary_result = boundary.reserve(
            1, JOB_A, ATTEMPT_A,
            quote(cap="0.1", reserved_rate="0.01", workload="10",
                  termination="10", retrieval="0", lag="0"), NOW)
        check("C6 Decimal ceiling equality admits without float drift",
              boundary_result.admitted
              and boundary_result.maximum_committed_usd == Decimal("0.3")
              and boundary_result.admission_limit_usd == Decimal("0.3"),
              boundary_result)

        # C7: a post-create price increase is durable exposure and demands
        # immediate termination.  A resource/tariff/deadline identity drift is
        # also fatal even when lower live compute masks its hourly total.  Only
        # a lower/equal live compute rate may differ from the reservation.
        increase = ready_ledger(root / "increase.json", ceiling="40")
        initial = increase.reserve(1, JOB_A, ATTEMPT_A,
                                   quote(cap="5", reserved_rate="1"), NOW)
        is_creating = increase.mark_creating(initial.generation, initial.attempt_key)
        increased = bind_observed_quote(
            increase, is_creating.generation, initial.attempt_key, "pod-rate-up",
            quote(cap="6", reserved_rate="1", live_rate="2"))
        frozen = increase.reserve(increased.generation, JOB_B, ATTEMPT_B,
                                  quote(cap="2"), NOW)
        increase_item = increase.snapshot()["attempts"][initial.attempt_key]

        drift = ready_ledger(root / "identity-drift.json", ceiling="40")
        drift_reserve = drift.reserve(1, JOB_A, ATTEMPT_A, quote(cap="5"), NOW)
        drift_creating = drift.mark_creating(
            drift_reserve.generation, drift_reserve.attempt_key)
        drift_bound = bind_observed_quote(
            drift, drift_creating.generation, drift_reserve.attempt_key,
            "pod-drift", quote(cap="5", live_rate="0.5", container_size="19"))

        lower = ready_ledger(root / "lower-live-rate.json", ceiling="40")
        lower_reserve = lower.reserve(1, JOB_A, ATTEMPT_A, quote(cap="5"), NOW)
        lower_creating = lower.mark_creating(
            lower_reserve.generation, lower_reserve.attempt_key)
        lower_bound = bind_observed_quote(
            lower, lower_creating.generation, lower_reserve.attempt_key,
            "pod-lower", quote(cap="5", live_rate="0.5"))
        check("C7 post-create rate/resource binding is fail-closed",
              increased.code == "TERMINATE_IMMEDIATELY"
              and increased.action == "TERMINATE_IMMEDIATELY"
              and increase_item["maximum_remaining_liability_usd"] == "6"
              and frozen.code == "ATTEMPT_FROZEN"
              and drift_bound.code == "TERMINATE_IMMEDIATELY"
              and lower_bound.code == "ACTUAL_QUOTE_BOUND"
              and lower_bound.action == "CONTINUE",
              (increased, increase_item, frozen, drift_bound, lower_bound))

        # C8: replace-before-crash state is complete and a fresh controller
        # sees the reservation with the same generation and Decimal amount.
        crash_path = root / "crash.json"
        before_crash = ready_ledger(crash_path, ceiling="30")
        crash_admit = before_crash.reserve(1, JOB_A, ATTEMPT_A,
                                           quote(cap="3.25"), NOW)
        del before_crash
        reopened = CampaignLedger(
            str(crash_path), PROVIDER, PROVIDER_ACCOUNT_ID).snapshot()
        check("C8 crash/reopen preserves a complete reservation",
              reopened["generation"] == crash_admit.generation
              and reopened["attempts"][crash_admit.attempt_key]
                  ["maximum_remaining_liability_usd"] == "3.25",
              reopened)

        # C9: deletion alone does not release and specifically freezes while
        # billing is unresolved.  Billing alone also does not release.  Only
        # both exact-ID proofs make remaining liability zero.
        release = ready_ledger(root / "release.json", ceiling="30")
        reserve = release.reserve(1, JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        creating_result = release.mark_creating(reserve.generation, reserve.attempt_key)
        actual = bind_observed_quote(
            release, creating_result.generation, reserve.attempt_key,
            "pod-release", quote(cap="4"))
        deleted = release.mark_deleted(actual.generation, reserve.attempt_key,
                                       "pod-release", NOW,
                                       "delete-response+exact-absence-sha256:test")
        after_delete = release.snapshot()["attempts"][reserve.attempt_key]
        blocked = release.reserve(deleted.generation, JOB_B, ATTEMPT_B,
                                  quote(cap="2"), NOW)
        billed = release.reconcile_billing(
            deleted.generation, reserve.attempt_key, "pod-release", NOW,
            Decimal("1.2345"), "billing-line-item-sha256:test")
        after_both = release.snapshot()["attempts"][reserve.attempt_key]
        released_inventory = observe_pods(release, billed.generation)
        next_admission = release.reserve(
            released_inventory.generation, JOB_B, ATTEMPT_B,
            quote(cap="2"), NOW)
        check("C9 release requires exact deletion and billing proof",
              deleted.applied
              and after_delete["maximum_remaining_liability_usd"] == "4"
              and not after_delete["released"]
              and blocked.code == "BILLING_UNRECONCILED"
              and billed.applied
              and after_both["maximum_remaining_liability_usd"] == "0"
              and after_both["released"]
              and after_both["phase"] == "RECONCILED"
              and next_admission.admitted,
              (deleted, after_delete, blocked, billed, after_both, next_admission))
        overcap = ready_ledger(root / "overcap.json", ceiling="30")
        overcap_reserved = overcap.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        overcap_creating = overcap.mark_creating(
            overcap_reserved.generation, overcap_reserved.attempt_key)
        overcap_bound = bind_observed_quote(
            overcap, overcap_creating.generation,
            overcap_reserved.attempt_key, "pod-overcap", quote(cap="4"))
        overcap_deleted = overcap.mark_deleted(
            overcap_bound.generation, overcap_reserved.attempt_key,
            "pod-overcap", NOW, "delete+absence-sha256:overcap")
        overcap_billing = overcap.reconcile_billing(
            overcap_deleted.generation, overcap_reserved.attempt_key,
            "pod-overcap", NOW, Decimal("4.01"),
            "billing-line-item-sha256:overcap")
        overcap_terminal = overcap.project_terminal_lease(
            overcap_deleted.generation, overcap_reserved.attempt_key,
            ["pod-overcap"], "lease-v2-self-seal:overcap",
            {"pod-overcap": {
                "deleted_at": NOW, "proof": "absence-sha256:overcap"}},
            NOW, Decimal("4.01"), "billing-line-item-sha256:overcap",
            provider_snapshot=terminal_provider_snapshot())
        overcap_after = overcap.snapshot()
        check("C9b charge above attempt hard cap cannot release liability",
              overcap_billing.code == "FINAL_CHARGE_EXCEEDS_HARD_CAP"
              and overcap_terminal.code == "FINAL_CHARGE_EXCEEDS_HARD_CAP"
              and overcap_after["generation"] == overcap_deleted.generation
              and overcap_after["attempts"][
                  overcap_reserved.attempt_key]["billing"] is None
              and overcap_after["attempts"][
                  overcap_reserved.attempt_key]["released"] is False,
              (overcap_billing, overcap_terminal, overcap_after))

        divergent = ready_ledger(
            root / "divergent-cap.json", ceiling="30")
        divergent_reserved = divergent.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        divergent_creating = divergent.mark_creating(
            divergent_reserved.generation, divergent_reserved.attempt_key)
        divergent_bound = bind_observed_quote(
            divergent, divergent_creating.generation,
            divergent_reserved.attempt_key, "pod-divergent",
            quote(cap="6"))
        divergent_deleted = divergent.mark_deleted(
            divergent_bound.generation, divergent_reserved.attempt_key,
            "pod-divergent", NOW, "delete+absence-sha256:divergent")
        divergent_billing = divergent.reconcile_billing_set(
            divergent_deleted.generation, divergent_reserved.attempt_key,
            ["pod-divergent"], NOW, Decimal("5"),
            "billing-line-item-sha256:divergent")
        divergent_terminal = divergent.project_terminal_lease(
            divergent_deleted.generation, divergent_reserved.attempt_key,
            ["pod-divergent"], "lease-v2-self-seal:divergent",
            {"pod-divergent": {
                "deleted_at": NOW, "proof": "absence-sha256:divergent"}},
            NOW, Decimal("5"), "billing-line-item-sha256:divergent",
            provider_snapshot=terminal_provider_snapshot())
        divergent_item = divergent.snapshot()["attempts"][
            divergent_reserved.attempt_key]
        check("C9c actual cap cannot raise reserved spending authority",
              divergent_bound.code == "TERMINATE_IMMEDIATELY"
              and divergent_billing.code
                  == "FINAL_CHARGE_EXCEEDS_HARD_CAP"
              and divergent_terminal.code
                  == "FINAL_CHARGE_EXCEEDS_HARD_CAP"
              and divergent_item["billing"] is None
              and divergent_item["released"] is False
              and divergent_item["maximum_remaining_liability_usd"] == "6",
              (divergent_bound, divergent_billing, divergent_terminal,
               divergent_item))


        # C10: network volume tariffs are represented, including hourly billing
        # granularity, but this first safe profile refuses the resource.
        network = ready_ledger(root / "network.json", ceiling="30")
        network_quote = quote(cap="2", network_size="10", network_rate="0.07")
        network_result = network.reserve(1, JOB_A, ATTEMPT_A, network_quote, NOW)
        check("C10 safe profile explicitly refuses network volume",
              network_result.code == "NETWORK_VOLUME_REFUSED"
              and network_quote.network_billing_increment_seconds == Decimal("3600")
              and network_quote.storage_month_hours == Decimal("672")
              and network_quote.tariff_source == RUNPOD_TARIFF_SOURCE,
              network_result)

        # C11: the provider deadline is an absolute backstop from create.  The
        # workload and retrieval reserve fit inside it rather than being added
        # to it a second time; only API/timer lag extends billable duration.
        backstop_quote = quote()
        invalid_backstop_refused = False
        try:
            quote(workload="600", retrieval="120", termination="719")
        except ValueError:
            invalid_backstop_refused = True
        check("C11 absolute provider deadline is not double-counted",
              backstop_quote.duration_seconds == Decimal("960")
              and invalid_backstop_refused,
              backstop_quote.duration_seconds)

        # C12: the campaign ceiling is lifetime spend, not merely concurrent
        # exposure.  Settled charge is added exactly once and attempt history
        # remains after release.
        sequential = ready_ledger(
            root / "sequential.json", ceiling="80", floor="0", margin="0",
            balance="1000")
        seq_reserved = sequential.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="60"), NOW)
        seq_creating = sequential.mark_creating(
            seq_reserved.generation, seq_reserved.attempt_key)
        seq_bound = bind_observed_quote(
            sequential, seq_creating.generation, seq_reserved.attempt_key,
            "pod-sequential", quote(cap="60"))
        seq_deleted = sequential.mark_deleted(
            seq_bound.generation, seq_reserved.attempt_key, "pod-sequential", NOW,
            "delete+absence-sha256:sequential")
        seq_billed = sequential.reconcile_billing(
            seq_deleted.generation, seq_reserved.attempt_key, "pod-sequential",
            NOW, Decimal("60"), "billing-line-sha256:sequential")
        duplicate_billing = sequential.reconcile_billing(
            seq_billed.generation, seq_reserved.attempt_key, "pod-sequential",
            NOW, Decimal("60"), "billing-line-sha256:sequential")
        sequential_inventory = observe_pods(
            sequential, seq_billed.generation)
        sequential_result = sequential.reserve(
            sequential_inventory.generation, JOB_B, ATTEMPT_B,
            quote(cap="25"), NOW)
        sequential_doc = sequential.snapshot()
        check("C12 settled charges enforce lifetime ceiling exactly once",
              seq_billed.applied
              and duplicate_billing.code == "BILLING_ALREADY_RECONCILED"
              and sequential_doc["settled_charges_usd"] == "60"
              and len(sequential_doc["attempts"]) == 1
              and sequential_result.code == "CEILING_EXCEEDED"
              and sequential_result.maximum_committed_usd == Decimal("85")
              and sequential_result.admission_limit_usd == Decimal("80"),
              (seq_billed, duplicate_billing, sequential_result, sequential_doc))

        # C13: public lifecycle transitions are monotonic.  Same-phase retries
        # are harmless/idempotent, while EXITED, TERMINATE_REQUIRED, deletion,
        # and reconciliation cannot be used to re-enter a chargeable run state.
        phases = ready_ledger(root / "phases.json", ceiling="30")
        phase_reserved = phases.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        phase_creating = phases.mark_creating(
            phase_reserved.generation, phase_reserved.attempt_key)
        phase_live = bind_observed_quote(
            phases, phase_creating.generation, phase_reserved.attempt_key,
            "pod-phases", quote(cap="4"))
        phase_running = phases.mark_phase(
            phase_live.generation, phase_reserved.attempt_key, "RUNNING")
        running_to_live = phases.mark_phase(
            phase_running.generation, phase_reserved.attempt_key, "LIVE")
        phase_exited = phases.mark_phase(
            phase_running.generation, phase_reserved.attempt_key, "EXITED")
        exited_to_running = phases.mark_phase(
            phase_exited.generation, phase_reserved.attempt_key, "RUNNING")
        phase_terminate = phases.mark_phase(
            phase_exited.generation, phase_reserved.attempt_key,
            "TERMINATE_REQUESTED")
        terminate_retry = phases.mark_phase(
            phase_terminate.generation, phase_reserved.attempt_key,
            "TERMINATE_REQUESTED")
        terminate_to_running = phases.mark_phase(
            phase_terminate.generation, phase_reserved.attempt_key, "RUNNING")
        deleted_phase = ready_ledger(root / "deleted-phase.json", ceiling="30")
        deleted_reserved = deleted_phase.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        deleted_creating = deleted_phase.mark_creating(
            deleted_reserved.generation, deleted_reserved.attempt_key)
        deleted_live = bind_observed_quote(
            deleted_phase, deleted_creating.generation,
            deleted_reserved.attempt_key, "pod-deleted-phase", quote(cap="4"))
        deleted_result = deleted_phase.mark_deleted(
            deleted_live.generation, deleted_reserved.attempt_key,
            "pod-deleted-phase", NOW, "delete+absence-sha256:phase")
        deleted_to_running = deleted_phase.mark_phase(
            deleted_result.generation, deleted_reserved.attempt_key, "RUNNING")
        required_to_live = increase.mark_phase(
            increased.generation, initial.attempt_key, "LIVE")
        reconciled_to_running = release.mark_phase(
            release.snapshot()["generation"], reserve.attempt_key, "RUNNING")
        check("C13 lifecycle graph refuses every regression",
              running_to_live.code == "INVALID_TRANSITION"
              and exited_to_running.code == "INVALID_TRANSITION"
              and terminate_to_running.code == "INVALID_TRANSITION"
              and deleted_to_running.code == "INVALID_TRANSITION"
              and required_to_live.code == "INVALID_TRANSITION"
              and reconciled_to_running.code == "ATTEMPT_RELEASED"
              and terminate_retry.applied
              and terminate_retry.code == "PHASE_UNCHANGED"
              and terminate_retry.generation == phase_terminate.generation
              and phases.snapshot()["attempts"][phase_reserved.attempt_key]
                  ["phase"] == "TERMINATE_REQUESTED",
              (running_to_live, exited_to_running, terminate_to_running,
               required_to_live, deleted_to_running, reconciled_to_running,
               terminate_retry))

        # C14: a controller may die after the provider returns an ID but before
        # scientific quote binding.  A reopened controller binds durable lease
        # identity for cleanup only; it can never RUN and exact-ID mismatch is
        # refused.  Ambiguous multiple IDs retain liability until every ID has
        # both absence and billing evidence.
        cleanup_path = root / "cleanup-bind.json"
        cleanup_ledger = ready_ledger(cleanup_path, ceiling="30")
        cleanup_reserved = cleanup_ledger.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        cleanup_creating = cleanup_ledger.mark_creating(
            cleanup_reserved.generation, cleanup_reserved.attempt_key)
        del cleanup_ledger
        reopened_cleanup = CampaignLedger(
            str(cleanup_path), PROVIDER, PROVIDER_ACCOUNT_ID)
        cleanup_bound = bind_observed_cleanup(
            reopened_cleanup, cleanup_creating.generation,
            cleanup_reserved.attempt_key, ["pod-after-crash"],
            "lease-v2-create-response-sha256:crash")
        cleanup_run = reopened_cleanup.mark_phase(
            cleanup_bound.generation, cleanup_reserved.attempt_key, "RUNNING")
        cleanup_rebind_mismatch = reopened_cleanup.bind_provider_for_cleanup(
            cleanup_bound.generation, cleanup_reserved.attempt_key,
            ["pod-other"], "lease-v2-create-response-sha256:other")
        cleanup_mismatch = reopened_cleanup.mark_deleted(
            cleanup_bound.generation, cleanup_reserved.attempt_key,
            "pod-wrong", NOW, "delete+absence-sha256:wrong")
        cleanup_deleted = reopened_cleanup.mark_deleted(
            cleanup_bound.generation, cleanup_reserved.attempt_key,
            "pod-after-crash", NOW, "delete+absence-sha256:crash")
        cleanup_billed = reopened_cleanup.reconcile_billing(
            cleanup_deleted.generation, cleanup_reserved.attempt_key,
            "pod-after-crash", NOW, Decimal("0.75"),
            "billing-line-sha256:crash")
        cleanup_item = reopened_cleanup.snapshot()["attempts"][
            cleanup_reserved.attempt_key]

        ambiguous = ready_ledger(root / "ambiguous-ids.json", ceiling="30")
        ambiguous_reserved = ambiguous.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        ambiguous_creating = ambiguous.mark_creating(
            ambiguous_reserved.generation, ambiguous_reserved.attempt_key)
        ambiguous_bound = bind_observed_cleanup(
            ambiguous, ambiguous_creating.generation,
            ambiguous_reserved.attempt_key, ["pod-z", "pod-a"],
            "lease-v2-ambiguous-create-sha256:test")
        ambiguous_delete_a = ambiguous.mark_deleted(
            ambiguous_bound.generation, ambiguous_reserved.attempt_key,
            "pod-a", NOW, "delete+absence-sha256:a")
        ambiguous_billing = ambiguous.reconcile_billing_set(
            ambiguous_delete_a.generation, ambiguous_reserved.attempt_key,
            ["pod-z", "pod-a"], NOW, Decimal("0.30"),
            "billing-reconciliation-sha256:all-ambiguous-ids")
        ambiguous_partial = ambiguous.snapshot()["attempts"][
            ambiguous_reserved.attempt_key]
        ambiguous_delete_z = ambiguous.mark_deleted(
            ambiguous_billing.generation, ambiguous_reserved.attempt_key,
            "pod-z", NOW, "delete+absence-sha256:z")
        ambiguous_final = ambiguous.snapshot()["attempts"][
            ambiguous_reserved.attempt_key]
        check("C14 crash cleanup binding projects every exact provider ID",
              cleanup_bound.code == "PROVIDER_BOUND_FOR_CLEANUP"
              and cleanup_bound.action == "TERMINATE_IMMEDIATELY"
              and cleanup_run.code == "INVALID_TRANSITION"
              and cleanup_rebind_mismatch.code == "CLEANUP_BINDING_MISMATCH"
              and cleanup_mismatch.code == "PROVIDER_ID_MISMATCH"
              and cleanup_billed.applied and cleanup_item["released"]
              and cleanup_item["actual_quote"] is None
              and cleanup_item["provider_ids"] == ["pod-after-crash"]
              and ambiguous_partial["provider_ids"] == ["pod-a", "pod-z"]
              and not ambiguous_partial["released"]
              and ambiguous_partial["maximum_remaining_liability_usd"] == "4"
              and ambiguous_billing.applied
              and ambiguous_delete_z.applied and ambiguous_final["released"],
              (cleanup_bound, cleanup_rebind_mismatch, cleanup_run,
               cleanup_mismatch, cleanup_item, ambiguous_partial,
               ambiguous_final))

        # C15: drill/dry planning reuses the exact admission decision under a
        # shared lock and cannot reserve width, liability, or generation.
        preview_ledger = ready_ledger(
            root / "preview.json", ceiling="30", width=1)
        preview_before = preview_ledger.snapshot()
        preview = preview_ledger.preview_reserve(
            preview_before["generation"], JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        preview_after = preview_ledger.snapshot()
        preview_real = preview_ledger.reserve(
            preview_before["generation"], JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        check("C15 admission preview is exact and mutation-free",
              preview.admitted and preview.code == "ADMISSIBLE"
              and preview.action == "NONE"
              and preview_before == preview_after
              and preview_real.admitted
              and preview_real.maximum_committed_usd
                  == preview.maximum_committed_usd,
              (preview, preview_before, preview_after, preview_real))

        # C16: lease PREPARED proves there is no fsynced POST intent, so a
        # campaign already marked CREATING can still release while retaining
        # auditable cancellation history.  Width becomes available again.
        prepared = ready_ledger(
            root / "prepared-cancel.json", ceiling="30", width=1)
        prepared_reserved = prepared.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        prepared_creating = prepared.mark_creating(
            prepared_reserved.generation, prepared_reserved.attempt_key)
        prepared_cancelled = prepared.cancel_before_create(
            prepared_creating.generation, prepared_reserved.attempt_key, NOW,
            "PREPARED", "lease-v2-prepared-self-seal:campaign-c16")
        prepared_retry = prepared.cancel_before_create(
            prepared_cancelled.generation, prepared_reserved.attempt_key, NOW,
            "PREPARED", "lease-v2-prepared-self-seal:campaign-c16")
        prepared_next = prepared.reserve(
            prepared_cancelled.generation, JOB_B, ATTEMPT_B, quote(cap="4"), NOW)
        prepared_item = prepared.snapshot()["attempts"][
            prepared_reserved.attempt_key]
        check("C16 campaign CREATING plus lease PREPARED cancels safely",
              prepared_cancelled.code == "CANCELLED_BEFORE_CREATE"
              and prepared_retry.code == "CANCELLATION_ALREADY_RECORDED"
              and prepared_item["released"]
              and prepared_item["phase"] == "CANCELLED_BEFORE_CREATE"
              and prepared_item["maximum_remaining_liability_usd"] == "0"
              and prepared_item["precreate_cancellation"]["lease_state"]
                  == "PREPARED"
              and prepared_item["precreate_cancellation"]
                  ["campaign_phase_before_cancel"] == "CREATING"
              and prepared_next.admitted,
              (prepared_cancelled, prepared_retry, prepared_item, prepared_next))

        # C17: after a controller crash, the systemd reaper can atomically bind
        # every exact lease ID and project complete absence + aggregate billing
        # evidence.  Matching retries neither re-add settled cost nor mutate.
        terminal = ready_ledger(root / "terminal-project.json", ceiling="30")
        terminal_reserved = terminal.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        terminal_creating = terminal.mark_creating(
            terminal_reserved.generation, terminal_reserved.attempt_key)
        terminal_absence = {
            "pod-crash-a": {
                "deleted_at": NOW,
                "proof": "provider-absence-sha256:pod-crash-a",
            },
            "pod-crash-z": {
                "deleted_at": NOW,
                "proof": "provider-absence-sha256:pod-crash-z",
            },
        }
        terminal_projected = terminal.project_terminal_lease(
            terminal_creating.generation, terminal_reserved.attempt_key,
            ["pod-crash-z", "pod-crash-a"],
            "lease-v2-self-seal:terminal-c17", terminal_absence,
            NOW, Decimal("1.25"),
            "billing-reconciliation-sha256:terminal-c17",
            provider_snapshot=terminal_provider_snapshot())
        terminal_retry = terminal.project_terminal_lease(
            terminal_projected.generation, terminal_reserved.attempt_key,
            ["pod-crash-z", "pod-crash-a"],
            "lease-v2-self-seal:terminal-c17", terminal_absence,
            NOW, Decimal("1.25"),
            "billing-reconciliation-sha256:terminal-c17",
            provider_snapshot=terminal_provider_snapshot())
        terminal_doc = terminal.snapshot()
        terminal_item = terminal_doc["attempts"][
            terminal_reserved.attempt_key]
        check("C17 terminal lease projection releases crash liability once",
              terminal_projected.code == "TERMINAL_LEASE_PROJECTED"
              and terminal_retry.code == "TERMINAL_LEASE_ALREADY_PROJECTED"
              and terminal_retry.generation == terminal_projected.generation
              and terminal_item["provider_ids"]
                  == ["pod-crash-a", "pod-crash-z"]
              and terminal_item["released"]
              and terminal_item["phase"] == "RECONCILED"
              and terminal_item["maximum_remaining_liability_usd"] == "0"
              and terminal_doc["settled_charges_usd"] == "1.25",
              (terminal_projected, terminal_retry, terminal_doc))
        staged = ready_ledger(
            root / "staged-controller-project.json", ceiling="30")
        staged_reserved = staged.reserve(
            1, JOB_B, ATTEMPT_B, quote(cap="4"), NOW)
        staged.mark_creating(
            staged_reserved.generation, staged_reserved.attempt_key)
        staged_lease = {
            "job_hash": JOB_B,
            "attempt_id": ATTEMPT_B,
            "create": {
                "provider": PROVIDER,
                "request_sha256": "c" * 64,
                "request": {
                    "campaign_ledger": "staged-controller-project.json",
                    "campaign_attempt_key": staged_reserved.attempt_key,
                    "provider": PROVIDER,
                    "provider_account_id": PROVIDER_ACCOUNT_ID,
                },
            },
            "provider_resource_ids": ["pod-after-crash"],
            "terminal_proof": {
                "provider_absence": {
                    "target_provider_ids": ["pod-after-crash"],
                    "still_present_ids": [],
                },
            },
            "billing_reconciliation": {
                "reconciled": True,
                "provider": PROVIDER,
                "provider_resource_ids": ["pod-after-crash"],
                "billing_histories": [{
                    "pod_id": "pod-after-crash",
                    "retrieved_at_utc": NOW,
                }],
                "total_amount": "0.75",
            },
            "history": [
                {
                    "event": "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING",
                    "to": "ABSENCE_CONFIRMED", "at": NOW,
                },
                {
                    "event": "BILLING_RECONCILIATION_STAGED_FOR_CAMPAIGN",
                    "to": "ABSENCE_CONFIRMED", "at": NOW,
                },
            ],
        }
        staged_binding = campaign_cleanup_binding_evidence(staged_lease)
        returned_id_binding = campaign_cleanup_binding_evidence(
            dict(staged_lease, provider_resource_ids=[]),
            {"pod-after-crash"})
        class EmptyFinalProvider:
            @staticmethod
            def chargeable_inventory():
                family = {
                    "complete": True, "resources": [], "source": "fixture"}
                return {
                    "schema": "fidelity-suite/runpod-chargeable-inventory.v1",
                    "provider": PROVIDER, "complete": True,
                    "unknown_families": [],
                    "observed_at_utc": "2026-09-01T12:00:04Z",
                    "families": {
                        "pods": dict(family),
                        "network_volumes": dict(family),
                    },
                }

            @staticmethod
            def list_lifecycle_resources():
                return []

            @staticmethod
            def status():
                return {
                    "id": PROVIDER_ACCOUNT_ID,
                    "clientBalance": "100",
                    "observed_at_utc": "2026-09-01T12:00:05Z",
                }

        finalize_campaign_after_absence(
            EmptyFinalProvider(), staged_lease,
            (root / "staged-leases").resolve())
        staged_item = staged.snapshot()["attempts"][
            staged_reserved.attempt_key]
        check("C17b controller and reaper use identical immutable cleanup binding",
              staged_item["released"]
              and staged_item["phase"] == "RECONCILED"
              and staged_item["billing"]["final_charge_usd"] == "0.75"
              and staged_item["cleanup_binding_evidence"] == staged_binding
              == returned_id_binding,
              staged_item)

        ordered = ready_ledger(
            root / "ordered-project.json", ceiling="30")
        ordered_reserved = ordered.reserve(
            1, JOB_A, ATTEMPT_A, quote(cap="4"), NOW)
        ordered.mark_creating(
            ordered_reserved.generation, ordered_reserved.attempt_key)
        ordered_lease = json.loads(json.dumps(staged_lease))
        ordered_lease["job_hash"] = JOB_A
        ordered_lease["attempt_id"] = ATTEMPT_A
        ordered_lease["create"]["request"]["campaign_ledger"] = (
            "ordered-project.json")
        ordered_lease["create"]["request"]["campaign_attempt_key"] = (
            ordered_reserved.attempt_key)

        original_commit = CampaignLedger._commit
        uncommitted_terminal = []

        def fail_terminal_commit(self, document):
            item = document["attempts"][ordered_reserved.attempt_key]
            uncommitted_terminal.append({
                "released": item["released"],
                "phase": item["phase"],
                "inventory": document["inventory"],
            })
            raise CampaignLedgerError("fixture atomic commit failure")

        CampaignLedger._commit = fail_terminal_commit
        snapshot_refused = False
        try:
            finalize_campaign_after_absence(
                EmptyFinalProvider(), ordered_lease,
                (root / "ordered-leases").resolve())
        except CampaignLedgerError:
            snapshot_refused = True
        finally:
            CampaignLedger._commit = original_commit
        ordered_item = ordered.snapshot()["attempts"][
            ordered_reserved.attempt_key]
        check("C17c snapshot and release share one atomic commit",
              snapshot_refused
              and len(uncommitted_terminal) == 1
              and uncommitted_terminal[0]["released"] is True
              and uncommitted_terminal[0]["phase"] == "RECONCILED"
              and uncommitted_terminal[0]["inventory"][
                  "provider_resources"] == []
              and ordered_item["released"] is False
              and ordered_item["phase"] == "CREATING"
              and ordered_item["maximum_remaining_liability_usd"] == "4",
              (uncommitted_terminal, ordered_item))

        # C18: spending state never follows symlinks or accepts loose ownership
        # boundaries.  Provider/account identity is immutable at open.
        hardening = root / "hardening"
        hardening.mkdir(mode=0o700)
        victim = hardening / "victim"
        victim.write_text("victim-unchanged")
        os.chmod(str(victim), 0o600)

        lock_symlink_path = hardening / "lock-symlink.json"
        (hardening / "lock-symlink.json.lock").symlink_to(victim)
        lock_symlink_refused = False
        try:
            CampaignLedger.create(
                str(lock_symlink_path), Decimal("10"), Decimal("1"),
                Decimal("1"), max_concurrent_attempts=1,
                provider=PROVIDER, provider_account_id=PROVIDER_ACCOUNT_ID)
        except CampaignLedgerError:
            lock_symlink_refused = True

        ledger_symlink_path = hardening / "ledger-symlink.json"
        ledger_symlink_path.symlink_to(victim)
        ledger_symlink_refused = False
        try:
            CampaignLedger.create(
                str(ledger_symlink_path), Decimal("10"), Decimal("1"),
                Decimal("1"), max_concurrent_attempts=1,
                provider=PROVIDER, provider_account_id=PROVIDER_ACCOUNT_ID)
        except CampaignLedgerError:
            ledger_symlink_refused = True

        loose_ledger_path = hardening / "loose-ledger.json"
        loose_ledger = ready_ledger(loose_ledger_path)
        secure_modes = (
            os.stat(str(loose_ledger_path)).st_mode & 0o777,
            os.stat(str(loose_ledger_path) + ".lock").st_mode & 0o777,
        )
        os.chmod(str(loose_ledger_path), 0o644)
        loose_ledger_refused = False
        try:
            loose_ledger.snapshot()
        except CampaignLedgerError:
            loose_ledger_refused = True

        loose_lock_path = hardening / "loose-lock.json"
        loose_lock = ready_ledger(loose_lock_path)
        os.chmod(str(loose_lock_path) + ".lock", 0o644)
        loose_lock_refused = False
        try:
            loose_lock.snapshot()
        except CampaignLedgerError:
            loose_lock_refused = True

        unsafe_parent = hardening / "group-writable"
        unsafe_parent.mkdir(mode=0o700)
        os.chmod(str(unsafe_parent), 0o770)
        unsafe_parent_refused = False
        try:
            CampaignLedger(
                str(unsafe_parent / "ledger.json"),
                PROVIDER, PROVIDER_ACCOUNT_ID)
        except CampaignLedgerError:
            unsafe_parent_refused = True

        account_path = hardening / "account.json"
        ready_ledger(account_path)
        account_swap_refused = False
        try:
            CampaignLedger(
                str(account_path), PROVIDER, "other-runpod-account").snapshot()
        except CampaignLedgerError:
            account_swap_refused = True

        check("C18 money ledger refuses symlinks, loose modes, and account swap",
              lock_symlink_refused and ledger_symlink_refused
              and victim.read_text() == "victim-unchanged"
              and secure_modes == (0o600, 0o600)
              and loose_ledger_refused and loose_lock_refused
              and unsafe_parent_refused and account_swap_refused,
              (lock_symlink_refused, ledger_symlink_refused, secure_modes,
               loose_ledger_refused, loose_lock_refused,
               unsafe_parent_refused, account_swap_refused))

        # C19: JSON decoding rejects duplicate object keys at every nesting
        # level instead of accepting the parser's last value.
        duplicate_cases = (
            ("top", '  "generation": 2,'),
            ("attempt", '      "phase": "RESERVED",'),
            ("quote", '        "hard_cap_usd": "2",'),
        )
        duplicate_results = []
        for suffix, needle in duplicate_cases:
            duplicate_path = hardening / ("duplicate-%s.json" % suffix)
            duplicate_ledger = ready_ledger(duplicate_path, ceiling="30")
            duplicate_reserved = duplicate_ledger.reserve(
                1, JOB_A, ATTEMPT_A, quote(cap="2"), NOW)
            if not duplicate_reserved.admitted:
                raise RuntimeError(duplicate_reserved)
            raw = duplicate_path.read_text()
            if raw.count(needle) != 1:
                raise RuntimeError("duplicate-key fixture needle is not unique")
            duplicate_path.write_text(
                raw.replace(needle, needle + "\n" + needle, 1))
            os.chmod(str(duplicate_path), 0o600)
            refused = False
            try:
                duplicate_ledger.snapshot()
            except CampaignLedgerError:
                refused = True
            duplicate_results.append(refused)
        check("C19 duplicate top-level, attempt, and quote keys are corrupt",
              all(duplicate_results), duplicate_results)

    if FAILED:
        print("\n%d campaign admission check(s) failed" % len(FAILED))
        return 1
    print("\nAll campaign admission checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
