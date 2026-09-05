#!/usr/bin/env python3
"""Selftest: the pre-spend arithmetic a human reads in one dry-run.

The defects (cloud usability review, 2026-09-05): the documented root recipe
refused on its own numbers and then two more times, one refusal per ~20 s
round trip (S1-2); `--max-runtime` was REQUIRED even for a target whose bound
is authored in bin/engines.json (S1-2); `--retrieval-delete-reserve` defaulted
to a flat 6 h that tipped a $4 candidate over its cap while every real run
passed the contract's own minimum by hand (S2-4); the plan printed the hard
cap as the only dollar figure and hid GPU, rate and datacenter in a 200 KB
JSON line (S2-3/S2-7); `reaper --list` was 96 filenames and `--sweep
--dry-run` printed nothing (S2-2).

Rungs (offline, $0.00, no provider or Hub access):
  R1  _defer_refusal + _raise_deferred_refusals: one finding raises itself;
      several raise one Refusal that numbers each reason and keeps every
      remedy line; nothing raises when none was deferred
  R2  _runpod_quote with deferred=list defers the cost refusal (returns an
      unbounded-cap quote and appends the finding); without the list it
      raises as before; the remedy names the figure to raise --max-cost to
  R3  parser: --max-runtime and --retrieval-delete-reserve default to None;
      _runpod_forbidden requires --max-runtime for quant only, and skips the
      reserve bound when it is to be derived
  R4  _root_workload_bound on the authored GLM-5.3 row is 26925 s for a fresh
      root on container-disk (the number the docs and the default carry)
  R5  reaper --list prints one block per unresolved lease with pod ids, ages,
      deadlines, last event, and for an AMBIGUOUS lease without ids the
      blockers and the operator sentence; terminal leases are a count; --all
      lists them; --sweep --dry-run prints every action row
  R6  _derive_index_census_allowlist (S1-3): with the Hub stubbed to the
      committed dy325 config/index bytes, the plan-time derivation binds
      inputs/allowlist.json with exactly the digests the authored table
      records for that pin (2d3aed81.../d567faf9..., 12311 names); a
      checkpoint with nothing past its boundary yields None; drifted bytes
      refuse
"""
import contextlib
import io
import json
import sys
import tempfile
import types
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import measure_cloud as MC  # noqa: E402
from fidelity.common import Console  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          ("  (%s)" % detail) if (detail and not ok) else ""))
    if not ok:
        failures.append(name)


def refusal_of(fn):
    try:
        fn()
    except MC.Refusal as exc:
        return exc
    return None


def main():
    # R1
    plan = {"_deferred_refusals": []}
    check("R1: nothing deferred, nothing raised",
          refusal_of(lambda: MC._raise_deferred_refusals(plan)) is None)
    MC._defer_refusal(plan, MC.Refusal("only one", ["fix it"]))
    one = refusal_of(lambda: MC._raise_deferred_refusals(plan))
    check("R1: one finding raises itself", one is not None and one.reason == "only one"
          and one.advice == ["fix it"])
    MC._defer_refusal(plan, MC.Refusal("second", ["a", "b"]))
    both = refusal_of(lambda: MC._raise_deferred_refusals(plan))
    text = "\n".join(both.advice) if both else ""
    check("R1: several findings raise one numbered report with every remedy",
          both is not None and "2 pre-spend findings" in both.reason
          and "[1] only one" in text and "[2] second" in text
          and "    fix it" in text and "    a" in text and "    b" in text, text)

    # R2
    args = types.SimpleNamespace(
        retrieval_delete_reserve=14400, timer_api_lag=600, max_cost="40",
        role="root", tariff_effective_at="2026-09-01T00:00:00Z",
        runpod_container_running_tariff="0.10", runpod_container_stopped_tariff="0.20",
        runpod_pod_running_tariff="0.10", runpod_pod_stopped_tariff="0.20",
        runpod_network_tariff="0.07")
    chosen = {"price_per_gpu_hour": "4.59", "gpus": 1}
    target = types.SimpleNamespace(repo_id="o/r", revision="a" * 40)
    timing = {"conservative_upper_hours": 8.0}
    warnings = []

    def quote(deferred):
        return MC._runpod_quote(args, chosen, target, "root-hf-transformers-bf16",
                                timing, 10, 1800, Decimal(27000), {},
                                warnings=warnings, deferred=deferred)
    raised = refusal_of(lambda: quote(None))
    check("R2: without a deferred list the cost refusal raises as before",
          raised is not None and raised.reason.startswith("all-in maximum $")
          and "exceeds --max-cost 40" in raised.reason
          and any("raise --max-cost to at least" in line for line in raised.advice),
          raised.reason if raised else "accepted")
    deferred = []
    with contextlib.redirect_stdout(io.StringIO()):
        priced = quote(deferred)
    check("R2: with the list the finding is deferred and a priced quote returns",
          len(deferred) == 1 and deferred[0].reason == raised.reason
          and priced.calculated_maximum_usd() > Decimal("40"))

    # R3
    parser = MC.build_parser()
    with contextlib.redirect_stderr(io.StringIO()):
        ns = parser.parse_args(["--provider", "runpod", "--model", "o/r"])
    check("R3: --max-runtime and --retrieval-delete-reserve default to None",
          ns.max_runtime is None and ns.retrieval_delete_reserve is None)
    base = dict(
        spot=False, region=None, on_preempt=None,
        dataset_id="fidelity--x.y.root.bf16", dataset_name=None,
        dataset_repository=None, publish_root_to="owner/repo",
        hf_token_file=__file__, hf_download_token_file=None,
        measurer="someone", cold_runs=2, max_cost="40",
        max_runtime=None, heartbeat_timeout=900,
        retrieval_delete_reserve=None, timer_api_lag=600,
        runpod_billing_wait=1800, sanity_expect="Paris",
        campaign_name="fidcloud-", campaign_ledger=None,
        campaign_ceiling=None, campaign_reserve=None,
        campaign_reaper_margin=None, runpod_safety_proof=None,
        campaign_width=1, width_two_root_archive=None,
        schedule="layer-outer", lane="streaming", capture_device="cuda",
        reduce_order="fp32", replay_device="numpy", replay_dtype="float32",
        replay_vocab_chunk=8192, form="hidden")
    root_args = types.SimpleNamespace(role="root", **base)
    root_forbidden = MC._runpod_forbidden(root_args)
    check("R3: root without --max-runtime / reserve passes the profile (derived later)",
          not any("max-runtime" in f or "retrieval-delete-reserve" in f
                  for f in root_forbidden), "; ".join(root_forbidden))
    quant_args = types.SimpleNamespace(role="quant", **dict(base, schedule="window-major"))
    quant_forbidden = MC._runpod_forbidden(quant_args)
    check("R3: quant without --max-runtime is still refused",
          "--max-runtime is required" in quant_forbidden, "; ".join(quant_forbidden))

    # R4
    from fidelity.engines import resolve_root_timing
    row = resolve_root_timing(
        target_repo="zai-org/GLM-5.3-BF16",
        target_revision="304b8051cfb2b260b61ce0cbe330e02a98e73639", gpu="H200",
        form="hidden", schedule="two-fresh-process-qualification")
    bound, derivation = MC._root_workload_bound(row, storage_layout="container-disk", captures=2)
    check("R4: the authored GLM-5.3 bound is 26925 s (components x 1.25)",
          int(bound) == 26925 and derivation["basis"] == "components_seconds"
          and derivation["fetch"] == "5520" and derivation["captures"] == 2, str(derivation))

    # R5: the reaper verbs over a stub store and provider.
    class Ref:
        def __init__(self, name):
            self.path = Path("/tmp") / name

    ambiguous = {
        "state": "AMBIGUOUS", "provider_resource_ids": [],
        "create": {"exact_name": "fidcloud-amb", "workload_deadline_utc": "2026-09-04T15:58:35Z",
                   "reap_deadline_utc": "2026-09-04T19:36:35Z"},
        "history": [{"at": "2026-09-04T14:58:35Z", "event": "LEASE_PREPARED_NO_PROVIDER_POST"},
                    {"at": "2026-09-04T15:08:53Z", "event": "LOST_CREATE_RESPONSE_RECONCILED_AMBIGUOUS"}],
        "terminal_proof": {"ambiguous_create": {"wrong_name_new_pod_ids": ["rix7"]}},
    }
    active = {
        "state": "ACTIVE", "provider_resource_ids": ["podactive"],
        "create": {"exact_name": "fidcloud-act", "workload_deadline_utc": "2026-09-05T17:22:19Z",
                   "reap_deadline_utc": "2026-09-05T21:12:37Z"},
        "history": [{"at": "2026-09-05T11:22:20Z", "event": "LEASE_PREPARED_NO_PROVIDER_POST"},
                    {"at": "2026-09-05T11:24:16Z", "event": "RESOURCE_IDENTITY_ATTESTED"}],
    }
    terminal = {"state": "TERMINAL", "provider_resource_ids": ["podgone"], "create": {}, "history": []}

    class Store:
        root = Path("/tmp/leases-stub")

        def list(self, *, include_terminal=True):
            return [(Ref("amb.json"), ambiguous), (Ref("act.json"), active),
                    (Ref("term.json"), terminal)]

    class Provider:
        def status(self):
            return {"id": "acct"}

        def chargeable_inventory(self):
            return {"complete": True, "families": {"pods": {"resources": [{"id": "podactive"}]}}}

    class Result:
        ok = True
        failures = []
        actions = [{"lease": "amb.json", "action": "ambiguous-needs-operator",
                    "blockers": {"wrong_name_new_pod_ids": ["rix7"]}},
                   {"lease": "act.json", "action": "would-reconcile-billing-and-campaign"}]

    import fidelity.cloudlease as CL
    originals = (CL.LeaseStore, CL.reap_once, CL.systemd_reaper_health)
    CL.LeaseStore = lambda path: Store()
    CL.reap_once = lambda store, providers, dry_run=False: Result()
    CL.systemd_reaper_health = lambda **kw: {"ok": True}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            def run(**flags):
                fields = dict(lease_dir=tmp, reaper_state_dir=tmp, install=False,
                              dry_run=False, list=False, sweep=False, runpod_key_file=None)
                fields.update(flags)
                ns = types.SimpleNamespace(**fields)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = MC._runpod_reaper_command(ns, Console(), Provider())
                return code, out.getvalue()
            code, text = run(list=True)
            check("R5: --list exits 0 and blocks the ACTIVE lease with its pod id, "
                  "presence, deadlines and last event",
                  code == 0 and "act.json  ACTIVE" in text
                  and "podactive" in text and "in inventory now" in text
                  and "2026-09-05T21:12:37Z" in text and "RESOURCE_IDENTITY_ATTESTED" in text,
                  text)
            check("R5: --list names the AMBIGUOUS lease's blockers and the operator act",
                  "AMBIGUOUS" in text and "rix7" in text
                  and "none exists in the account inventory" in text
                  and "needs operator" in text and "fidcloud-amb" in text
                  and "--allow-unresolved-leases" in text, text)
            check("R5: --list counts terminal leases instead of listing them",
                  "1 lease(s) settled" in text and "TERMINAL" not in text)
            check("R5: --list shows ages, not just stamps", " ago)" in text)
            code, text_all = run(list=True, all=True)
            check("R5: --list --all lists terminal leases", "TERMINAL" in text_all)
            code, sweep = run(sweep=True, dry_run=True)
            check("R5: --sweep --dry-run prints every action row",
                  code == 0 and "ambiguous-needs-operator" in sweep
                  and "would-reconcile-billing-and-campaign" in sweep and "rix7" in sweep, sweep)
    finally:
        CL.LeaseStore, CL.reap_once, CL.systemd_reaper_health = originals

    # R6: plan-time derivation against the committed dy325 allowlist.
    from fidelity.runpodsafety import _ALLOWLISTS
    pin = ("davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw", "6d6bd738c0c1635513e0bd0fdf0302049bd820a9")
    authored = _ALLOWLISTS[pin]
    evidence = ROOT / "engines" / "tools" / "layer-outer-evidence"
    sidecar = json.loads((evidence / "dy325-exl3-layer78-unexpected-keys.json.provenance.json")
                         .read_text("utf-8"))
    index_doc = {"metadata": {}, "weight_map": {}}
    for name in json.loads((evidence / "dy325-exl3-layer78-unexpected-keys.json").read_text("utf-8")):
        index_doc["weight_map"][name] = "model-00001.safetensors"
    index_doc["weight_map"]["model.layers.0.mlp.down_proj.weight"] = "model-00001.safetensors"
    index_raw = json.dumps(index_doc).encode()
    config_raw = json.dumps({"architectures": ["GlmMoeDsaForCausalLM"],
                             "num_hidden_layers": 78}).encode()
    import hashlib
    files = {"config.json": config_raw, "model.safetensors.index.json": index_raw}
    original_fetch = MC.fetch_file
    MC.fetch_file = lambda repo, name, revision=None, **kw: files[name]
    target = types.SimpleNamespace(repo_id=pin[0], revision=pin[1])
    identity = {"config_sha256": hashlib.sha256(config_raw).hexdigest(),
                "index_sha256": hashlib.sha256(index_raw).hexdigest()}
    try:
        plan = {"warnings": []}
        with contextlib.redirect_stdout(io.StringIO()):
            derived = MC._derive_index_census_allowlist(target, identity, plan, Console())
        check("R6: the derivation reproduces the authored dy325 digests from the index",
              derived is not None
              and derived["artifact_sha256"] == authored["artifact_sha256"]
              and derived["canonical_sorted_names_sha256"] == authored["canonical_sorted_names_sha256"]
              and derived["count"] == authored["count"] == sidecar["count"]
              and derived["path"] == "inputs/allowlist.json"
              and Path(plan["_derived_allowlist_local"]).is_file(),
              repr(derived))
        files["model.safetensors.index.json"] = json.dumps(
            {"weight_map": {"model.layers.0.a": "x"}}).encode()
        identity["index_sha256"] = hashlib.sha256(files["model.safetensors.index.json"]).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            none = MC._derive_index_census_allowlist(target, identity, {"warnings": []}, Console())
        check("R6: nothing past the boundary derives None (no allowlist bound)", none is None)
        identity["index_sha256"] = "0" * 64
        drift = refusal_of(lambda: MC._derive_index_census_allowlist(
            target, identity, {"warnings": []}, Console()))
        check("R6: bytes that differ from the identity refuse",
              drift is not None and "differ from the identity" in drift.reason)
    finally:
        MC.fetch_file = original_fetch

    print()
    if failures:
        print("selftest_measure_cloud_prespend: %d FAILED" % len(failures))
        return 1
    print("selftest_measure_cloud_prespend: all rungs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
