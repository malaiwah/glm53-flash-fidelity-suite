#!/usr/bin/env python3
"""Validate the quant-fidelity-registry: schema conformance PLUS the invariants
that make the comparability guarantee true rather than aspirational.

Runs OFFLINE with no installs. Sources are shape-checked; they are never fetched.

  python3 tools/registry_validate.py [--root DIR] [--strict] [--json]
                                     [--only CHECK[,CHECK...]] [--skip CHECK[,...]]
                                     [--jsonschema-lib mini|external|both]
                                     [--explain MEAS_ID [--against MEAS_ID]]
                                     [--offline-selftest]

Exit: 0 clean - 1 errors - 2 warnings only (under --strict) - 4 internal error
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry_lib as L      # noqa: E402
import _minischema            # noqa: E402

FORBIDDEN_NET_MODULES = ("socket", "ssl", "urllib", "http", "requests", "huggingface_hub",
                         "aiohttp", "httpx", "ftplib", "telnetlib", "xmlrpc")


class Report(object):
    def __init__(self):
        self.findings = []

    def add(self, check, severity, message, rid=None, remedy=None):
        self.findings.append({"check": check, "severity": severity, "id": rid,
                              "message": message, "remedy": remedy})

    def err(self, check, message, rid=None, remedy=None):
        self.add(check, "error", message, rid, remedy)

    def warn(self, check, message, rid=None, remedy=None):
        self.add(check, "warn", message, rid, remedy)

    @property
    def errors(self):
        return [f for f in self.findings if f["severity"] == "error"]

    @property
    def warnings(self):
        return [f for f in self.findings if f["severity"] == "warn"]


# ---------------------------------------------------------------------------
# L0/L1 - format and schema
# ---------------------------------------------------------------------------

def check_format_and_schema(root, rep, lib):
    data_dir = os.path.join(root, "data")
    schema_dir = os.path.join(root, "schema")
    reg = None
    if lib in ("mini", "both"):
        reg = _minischema.Registry(schema_dir)
    ext = None
    if lib in ("external", "both"):
        ext = _external_validator(schema_dir, rep)

    collections = {}
    for name, tag, schema_file in L.COLLECTIONS:
        path = os.path.join(data_dir, name + ".jsonl")
        if not os.path.exists(path):
            rep.err("L0.MISSING", "collection file not found: %s" % path)
            collections[name] = {}
            continue
        rows, prev_id = [], None
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except ValueError as exc:
                    rep.err("L0.PARSE", "%s:%d not valid JSON: %s" % (name, lineno, exc))
                    continue
                if L.canonical_json(obj) != line:
                    rep.err("L0.CANONICAL", "%s:%d is not canonical JSON" % (name, lineno),
                            obj.get("id"), "run tools/seed_registry.py or registry_add.py to rewrite it")
                rid = obj.get("id")
                if not isinstance(rid, str) or not L.ID_RE.match(rid or ""):
                    rep.err("L0.ID_GRAMMAR", "%s:%d id %r does not match the registry id grammar"
                            % (name, lineno, rid))
                elif L.collection_of_id(rid) != name:
                    rep.err("L0.ID_GRAMMAR", "%s:%d id %r has the wrong collection prefix"
                            % (name, lineno, rid), rid)
                if prev_id is not None and rid is not None and rid < prev_id:
                    rep.err("L0.SORTED", "%s:%d id %r sorts before the previous line" % (name, lineno, rid), rid)
                prev_id = rid
                rows.append(obj)
                if reg is not None:
                    for e in reg.validate(obj, schema_file):
                        rep.err("L1.SCHEMA", "%s[%s]%s" % (name, rid, e), rid)
                if ext is not None:
                    for msg in ext(obj, schema_file):
                        rep.err("L1.SCHEMA_EXT", "%s[%s] %s" % (name, rid, msg), rid)
        seen = {}
        for obj in rows:
            rid = obj.get("id")
            if rid in seen:
                rep.err("L0.ID_UNIQUE", "duplicate id %r in %s" % (rid, name), rid)
            seen[rid] = obj
        collections[name] = seen

    all_ids = {}
    for name, recs in collections.items():
        for rid in recs:
            if rid in all_ids:
                rep.err("L0.ID_UNIQUE", "id %r appears in both %s and %s" % (rid, all_ids[rid], name), rid)
            all_ids[rid] = name
    return collections


def _external_validator(schema_dir, rep):
    try:
        import jsonschema
        from referencing import Registry as RRegistry, Resource
        from referencing.jsonschema import DRAFT202012
    except Exception:
        return None
    resources = []
    for fn in sorted(os.listdir(schema_dir)):
        if fn.endswith(".schema.json"):
            with open(os.path.join(schema_dir, fn), encoding="utf-8") as fh:
                doc = json.load(fh)
            resources.append((fn, Resource(contents=doc, specification=DRAFT202012)))
    registry = RRegistry().with_resources(resources)

    def run(obj, schema_file):
        with open(os.path.join(schema_dir, schema_file), encoding="utf-8") as fh:
            schema = json.load(fh)
        v = jsonschema.Draft202012Validator(schema, registry=registry)
        return ["%s: %s" % ("/".join(str(p) for p in e.absolute_path) or "<root>", e.message)
                for e in v.iter_errors(obj)]
    return run


# ---------------------------------------------------------------------------
# L2 - referential integrity  (REF-*)
# ---------------------------------------------------------------------------

def _ref_ok(rep, check, rid, field, value, coll, C, optional=False):
    if value is None:
        if not optional:
            rep.err(check, "%s.%s is null" % (rid, field), rid)
        return None
    if L.collection_of_id(value) != coll:
        rep.err(check, "%s.%s = %r is not a %s id" % (rid, field, value, coll), rid)
        return None
    if value not in C[coll]:
        rep.err(check, "%s.%s = %r does not resolve in %s" % (rid, field, value, coll), rid,
                "add the record or fix the reference")
        return None
    return C[coll][value]


def check_referential(C, rep):
    for aid, a in C["artifacts"].items():
        _ref_ok(rep, "REF-001", aid, "model_ref", a.get("model_ref"), "models", C)
        if a.get("derived_from_artifact_ref"):
            _ref_ok(rep, "REF-001", aid, "derived_from_artifact_ref", a["derived_from_artifact_ref"],
                    "artifacts", C)
    for mid, m in C["models"].items():
        _ref_ok(rep, "REF-001", mid, "canonical_weights.artifact_ref",
                (m.get("canonical_weights") or {}).get("artifact_ref"), "artifacts", C)
    for pid, p in C["panels"].items():
        for mref in p.get("model_scope", []):
            _ref_ok(rep, "REF-001", pid, "model_scope", mref, "models", C)
        if p.get("derived_from"):
            _ref_ok(rep, "REF-008", pid, "derived_from", p["derived_from"], "panels", C)
    for rid, r in C["references"].items():
        _ref_ok(rep, "REF-001", rid, "artifact_ref", r.get("artifact_ref"), "artifacts", C)
        _ref_ok(rep, "REF-001", rid, "panel_ref", r.get("panel_ref"), "panels", C)
        fm = (r.get("self_consistency") or {}).get("floor_measurement_ref")
        if fm:
            _ref_ok(rep, "REF-010", rid, "self_consistency.floor_measurement_ref", fm, "measurements", C)

    # A pipeline that declares a lane may name the sealed measurement it was bridged
    # against. That reference is the whole evidentiary weight of the bridge -- a dangling
    # one would leave a delta hanging off nothing.
    for pid, pl in C["pipelines"].items():
        br = ((pl.get("lane") or {}).get("bridge") or {})
        if br.get("panel_ref"):
            _ref_ok(rep, "REF-001", pid, "lane.bridge.panel_ref", br["panel_ref"], "panels", C)
        if br.get("sealed_measurement_ref"):
            _ref_ok(rep, "REF-010", pid, "lane.bridge.sealed_measurement_ref",
                    br["sealed_measurement_ref"], "measurements", C)

    for mid, m in C["measurements"].items():
        art = _ref_ok(rep, "REF-001", mid, "artifact_ref", m.get("artifact_ref"), "artifacts", C)
        pan = _ref_ok(rep, "REF-001", mid, "panel_ref", m.get("panel_ref"), "panels", C)
        ref = _ref_ok(rep, "REF-001", mid, "reference_ref", m.get("reference_ref"), "references", C)
        _ref_ok(rep, "REF-001", mid, "pipeline_ref", m.get("pipeline_ref"), "pipelines", C)
        _ref_ok(rep, "REF-001", mid, "model_ref", m.get("model_ref"), "models", C)
        if art and art.get("model_ref") != m.get("model_ref"):
            rep.err("REF-003", "%s.model_ref %s != its artifact's model_ref %s"
                    % (mid, m.get("model_ref"), art.get("model_ref")), mid)
        if ref and ref.get("panel_ref") != m.get("panel_ref"):
            rep.err("REF-004", "%s: the reference was captured on panel %s but the row claims %s. "
                                "A teacher captured on another panel can never back a row."
                    % (mid, ref.get("panel_ref"), m.get("panel_ref")), mid)
        if ref:
            rart = C["artifacts"].get(ref.get("artifact_ref"))
            if rart and rart.get("model_ref") != m.get("model_ref") and not L.has_disclosure(
                    ref, "cross_model_reference"):
                rep.err("REF-005", "%s: the reference's artifact belongs to model %s, the row to %s, and the "
                                    "reference does not disclose cross_model_reference"
                        % (mid, rart.get("model_ref"), m.get("model_ref")), mid)
        if pan and m.get("model_ref") not in pan.get("model_scope", []):
            rep.err("REF-006", "%s: panel %s does not list model %s in model_scope"
                    % (mid, m.get("panel_ref"), m.get("model_ref")), mid)
        for field, coll in (("supersedes", "measurements"),):
            if m.get(field):
                _ref_ok(rep, "REF-010", mid, field, m[field], coll, C)
                if m[field] == mid:
                    rep.err("REF-010", "%s.%s is self-referential" % (mid, field), mid)
        bias = (m.get("comparability") or {}).get("bias")
        if bias and bias.get("floor_measurement_ref"):
            fm = bias["floor_measurement_ref"]
            _ref_ok(rep, "REF-010", mid, "comparability.bias.floor_measurement_ref", fm, "measurements", C)
            if fm == mid:
                rep.err("REF-010", "%s names itself as its own floor" % mid, mid)
        ver = (m.get("provenance") or {}).get("verification")
        if ver and ver.get("verification_measurement_ref"):
            _ref_ok(rep, "REF-010", mid, "provenance.verification.verification_measurement_ref",
                    ver["verification_measurement_ref"], "measurements", C)

    # tokenizer binding (REF-007)
    for pid, p in C["panels"].items():
        ptok = (p.get("tokenizer") or {}).get("id")
        for mref in p.get("model_scope", []):
            mdl = C["models"].get(mref)
            if mdl and (mdl.get("tokenizer") or {}).get("id") != ptok:
                rep.err("REF-007", "panel %s uses tokenizer %r but model %s declares %r"
                        % (pid, ptok, mref, (mdl.get("tokenizer") or {}).get("id")), pid)

    # cycles (REF-008 / REF-009)
    _no_cycles(C["panels"], "derived_from", rep, "REF-008")
    _no_cycles(C["artifacts"], "derived_from_artifact_ref", rep, "REF-009")

    # unreferenced records (informational)
    used = set()
    for m in C["measurements"].values():
        used.update({m.get("panel_ref"), m.get("reference_ref"), m.get("pipeline_ref"),
                     m.get("artifact_ref")})
    for coll in ("panels", "references", "pipelines"):
        for rid in C[coll]:
            if rid not in used:
                rep.warn("L2.UNREFERENCED", "%s is not used by any measurement" % rid, rid)


def _no_cycles(records, field, rep, check):
    for start in records:
        seen, cur = set(), start
        while cur:
            if cur in seen:
                rep.err(check, "cycle in %s starting at %s" % (field, start), start)
                break
            seen.add(cur)
            cur = (records.get(cur) or {}).get(field)


# ---------------------------------------------------------------------------
# L3 - comparability (CMP-*, BIAS-*, PANEL-007)
# ---------------------------------------------------------------------------

def check_comparability(C, rep):
    groups = {}
    for mid, m in C["measurements"].items():
        comp = m.get("comparability") or {}
        derived = L.key_inputs_from_measurement(m)
        declared = comp.get("key_inputs") or {}
        for f in L.COMPARABILITY_KEY_FIELDS:
            if declared.get(f) != derived[f]:
                rep.err("CMP-002", "%s: comparability.key_inputs.%s = %r but the row's own fields say %r"
                        % (mid, f, declared.get(f), derived[f]), mid,
                        "key_inputs is an expansion of the row, never an override")
        want = L.comparability_key(derived)
        if comp.get("key") != want:
            rep.err("CMP-001", "%s: comparability.key %s does not match the key recomputed from the row (%s). "
                                "A hand-written key cannot move a number into another table."
                    % (mid, comp.get("key"), want), mid)
        groups.setdefault(want, []).append(mid)

        pan = C["panels"].get(m.get("panel_ref")) or {}
        ms = m.get("measurement_scope") or {}
        total = (pan.get("structure") or {}).get("scored_positions_total")
        if ms.get("covers_full_panel"):
            if total is not None and ms.get("scored_positions") != total:
                rep.err("SCOPE-007", "%s claims to cover the full panel but scores %s of %s positions"
                        % (mid, ms.get("scored_positions"), total), mid)
            pc = (pan.get("structure") or {}).get("contexts")
            if pc is not None and ms.get("contexts") is not None and ms["contexts"] != pc:
                rep.err("SCOPE-007", "%s claims full panel coverage with %s contexts, panel has %s"
                        % (mid, ms["contexts"], pc), mid)

        if comp.get("class") == "strict":
            if not pan.get("sealed"):
                rep.err("PANEL-007", "%s is comparability.class=strict on an unsealed panel %s"
                        % (mid, m.get("panel_ref")), mid)
            if not pan.get("contamination", {}).get("checked"):
                rep.warn("PANEL-006", "%s is strict on panel %s whose contamination.checked is false"
                         % (mid, m.get("panel_ref")), mid)
            art = C["artifacts"].get(m.get("artifact_ref")) or {}
            if any(a.get("treatment") == "unknown" for a in (art.get("scope") or {}).get("assignments", [])):
                rep.err("SCOPE-009", "%s is strict but its artifact has unknown scope assignments" % mid, mid)
            if comp.get("bias"):
                rep.err("BIAS-003", "%s is strict but carries a bias block" % mid, mid)

        # BIAS-001 / 002 / 004
        est = m.get("estimator") or {}
        bias = comp.get("bias")
        if est.get("stack_relation") == "cross_stack":
            if not bias:
                rep.err("BIAS-001", "%s is cross_stack with no bias block. A cross-stack number without its "
                                     "floor is not publishable here." % mid, mid)
            else:
                if bias.get("kind") != "cross_stack_capture_replay":
                    rep.err("BIAS-001", "%s: cross_stack requires bias.kind=cross_stack_capture_replay" % mid, mid)
                if bias.get("direction") == "unknown":
                    rep.err("BIAS-001", "%s: cross_stack bias.direction must not be unknown" % mid, mid)
                if not bias.get("floor_measurement_ref") and len(bias.get("detail", "")) < 40:
                    rep.err("BIAS-001", "%s: no floor_measurement_ref and no explicit detail saying why "
                                         "no floor exists" % mid, mid)
        if bias and bias.get("floor_measurement_ref"):
            floor = C["measurements"].get(bias["floor_measurement_ref"])
            if floor:
                fk = (floor.get("comparability") or {}).get("key")
                if fk != comp.get("key"):
                    rep.err("BIAS-002", "%s: its floor %s has comparability key %s, not %s. A floor from a "
                                         "different panel or estimator is not a floor."
                            % (mid, bias["floor_measurement_ref"], fk, comp.get("key")), mid)
                ref = C["references"].get(m.get("reference_ref")) or {}
                if not _is_floor_artifact(C, floor.get("artifact_ref"), ref.get("artifact_ref")):
                    rep.err("BIAS-004", "%s: its floor measures %s, which is neither the reference's own "
                                         "artifact %s nor an unquantized repack of it. A floor measures "
                                         "UNQUANTIZED weights through the candidate's stack."
                            % (mid, floor.get("artifact_ref"), ref.get("artifact_ref")), mid)

    # CMP-003 / CMP-005
    for key, members in sorted(groups.items()):
        positions = {}
        for mid in members:
            ms = C["measurements"][mid].get("measurement_scope") or {}
            positions.setdefault(ms.get("scored_positions"), []).append(mid)
        if len(positions) > 1:
            for pos, mids in positions.items():
                for mid in mids:
                    ms = C["measurements"][mid].get("measurement_scope") or {}
                    if ms.get("covers_full_panel"):
                        rep.err("CMP-003", "%s shares comparability key %s with rows scoring a different "
                                            "number of positions, yet claims full panel coverage"
                                % (mid, key), mid)
        if len(members) == 1:
            rep.warn("CMP-005", "comparability key %s has a single member (%s): a number with nothing to "
                                "compare against. Present it as a stated fact, not a ranking."
                     % (key, members[0]), members[0])

    # CMP-004
    seen = {}
    for mid, m in sorted(C["measurements"].items()):
        est = m.get("estimator") or {}
        tup = (m.get("artifact_ref"), m.get("panel_ref"), m.get("reference_ref"), m.get("pipeline_ref"),
               (m.get("metric") or {}).get("name"), est.get("stack_relation"), est.get("head_policy"),
               (m.get("provenance") or {}).get("measured_by"))
        if tup in seen:
            other = seen[tup]
            if m.get("status") not in ("superseded", "retracted") and \
               C["measurements"][other].get("status") not in ("superseded", "retracted"):
                rep.err("CMP-004", "%s and %s are the same (artifact, panel, reference, pipeline, metric, "
                                    "stack_relation, head_policy, measured_by) and neither is superseded"
                        % (mid, other), mid)
        seen[tup] = mid
    return groups


# ---------------------------------------------------------------------------
# L4 - provenance (PROV-*)
# ---------------------------------------------------------------------------

HASHED_SOURCE_KINDS = ("receipt_file", "hf_file", "github_file")


def _owner(uri):
    """The namespace a source URI belongs to, for attribution checks only.

    Purely lexical: this never fetches anything (OFFLINE-001). A local filesystem
    path is 'local', anything we cannot parse is 'unattributed'.
    """
    u = (uri or "").strip()
    if not u:
        return "unattributed"
    # A sealed submission receipt lives in-repo at receipts/<handle>/<slug>.json, so the
    # directory IS the attribution -- that is the whole point of the contributor layout.
    if u.startswith("receipts/"):
        parts = u.split("/")
        return parts[1] if len(parts) > 2 and parts[1] else "unattributed"
    for host, offset in (("huggingface.co/datasets/", 0), ("huggingface.co/", 0),
                         ("raw.githubusercontent.com/", 0), ("github.com/", 0)):
        i = u.find(host)
        if i >= 0:
            rest = u[i + len(host):].split("/")
            return rest[offset] if rest and rest[offset] else "unattributed"
    if "://" in u:
        return "unattributed"
    # Everything left is a filesystem path: absolute, in-repo relative, or one of the
    # prose placeholders the seeder writes ("scratchpad copy of ..."). A file we can
    # open is a file we hold.
    return "local"


def _ours(uri):
    """True when the URI is a receipt this registry's maintainer actually holds."""
    owner = _owner(uri)
    if owner == "local":
        return True
    return owner.lower().startswith(L.MAINTAINER.lower())


def _is_floor_artifact(C, floor_aid, ref_aid):
    """A floor measures unquantized weights: the reference's own artifact, or a base-kind
    artifact that derives from it (a cross-ENGINE floor necessarily uses the other engine's
    repack of the same base weights, e.g. a BF16 .gguf)."""
    if floor_aid == ref_aid:
        return True
    a = C["artifacts"].get(floor_aid) or {}
    if a.get("kind") != "base":
        return False
    seen, cur = set(), floor_aid
    while cur and cur not in seen:
        seen.add(cur)
        if cur == ref_aid:
            return True
        cur = (C["artifacts"].get(cur) or {}).get("derived_from_artifact_ref")
    return False


def check_provenance(C, rep):
    for mid, m in C["measurements"].items():
        pv = m.get("provenance") or {}
        by = pv.get("measured_by")
        measurer = pv.get("measurer") or {}
        sources = pv.get("sources") or []
        hashed = [s for s in sources if s.get("kind") in HASHED_SOURCE_KINDS and s.get("sha256")]
        if by == "self-measured":
            if not measurer.get("is_registry_maintainer"):
                rep.err("PROV-001", "%s is self-measured but the measurer is not the registry maintainer" % mid, mid)
            if not hashed:
                rep.err("PROV-001", "%s is self-measured with no hashed receipt/hf/github source. You cannot "
                                     "claim to have measured something without holding a hashed receipt." % mid,
                        mid, "attach the receipt with its sha256, or set measured_by to a reported value")
            # PROV-010. Being self-measured means we produced the receipt, so at least one
            # source must be a receipt WE hold: a local path, or a URI in our own namespace.
            # A row whose entire evidence lives in somebody else's repository is that
            # person's measurement wearing our name, which is the failure mode PROV-007
            # only catches while the author_reported_only disclosure is still attached.
            if hashed and not any(_ours(s.get("uri") or "") for s in hashed):
                rep.err("PROV-010",
                        "%s is self-measured, but every one of its %d hashed sources lives in someone "
                        "else's namespace (%s). A number we claim to have produced must cite at least "
                        "one receipt we hold."
                        % (mid, len(hashed),
                           ", ".join(sorted({_owner(s.get("uri") or "") for s in hashed}))),
                        mid,
                        "publish our own receipt and cite it, or set measured_by to author-reported / "
                        "third-party-reported and credit whoever produced the number")
        else:
            if measurer.get("is_registry_maintainer"):
                rep.err("PROV-002", "%s is %s but names the registry maintainer as measurer" % (mid, by), mid)
            if (m.get("comparability") or {}).get("class") != "advisory":
                rep.err("PROV-002", "%s is %s and must be comparability.class=advisory" % (mid, by), mid)
            if not L.has_disclosure(m, "author_reported_only"):
                rep.err("PROV-002", "%s is %s without an author_reported_only disclosure" % (mid, by), mid)
        if pv.get("independently_verified"):
            ver = pv.get("verification")
            if not ver:
                rep.err("PROV-003", "%s claims independent verification with no verification block" % mid, mid)
            else:
                vb = ver.get("verified_by") or {}
                if vb.get("name") == measurer.get("name"):
                    rep.err("PROV-003", "%s: the verifier (%s) is the same party as the measurer. "
                                         "Verification means somebody else reproduced it."
                            % (mid, vb.get("name")), mid)
            if not any(s.get("kind") in HASHED_SOURCE_KINDS or
                       (s.get("kind") == "url" and s.get("sha256")) for s in sources):
                rep.err("PROV-004", "%s claims verification but has no verifiable source" % mid, mid)
        if L.has_disclosure(m, "author_reported_only") and by == "self-measured":
            rep.err("PROV-007", "%s is marked self-measured yet carries an author_reported_only "
                                 "disclosure. Somebody else's measurement cannot be relabelled as ours."
                    % mid, mid)
        pl = C["pipelines"].get(m.get("pipeline_ref")) or {}
        pl_author = (pl.get("author") or {}).get("name")
        if by == "self-measured" and pl_author and pl_author != measurer.get("name"):
            rep.err("PROV-008", "%s is self-measured but ran on pipeline %s, which is authored by %s. "
                                 "A row we measured is a row that ran on our stack."
                    % (mid, m.get("pipeline_ref"), pl_author), mid)
        for s in sources:
            if s.get("kind") == "receipt_file" and (s.get("uri", "").startswith("/")):
                mirrors = [x for x in sources if x.get("kind") in ("hf_file", "github_file")]
                if not mirrors:
                    rep.warn("PROV-005", "%s rests on a receipt only we can see (%s) with no published mirror. "
                                          "A receipt nobody can fetch is a receipt nobody can check."
                             % (mid, s.get("uri")), mid)
                break

    # PROV-012 - the lane is a property of the row, not a footnote on the pipeline
    for mid, m in sorted(C["measurements"].items()):
        pl = C["pipelines"].get(m.get("pipeline_ref")) or {}
        lane = (pl.get("lane") or {}).get("name")
        if not lane or lane == "sealed-ep8":
            continue
        if not L.has_disclosure(m, "non_sealed_lane", affects=True):
            rep.err("PROV-012",
                    "%s ran on pipeline %s, whose declared lane is %r, but the row carries no "
                    "non_sealed_lane disclosure with affects_comparability=true. The comparability "
                    "key has no lane input, so this row is tabled beside sealed-lane rows with "
                    "nothing on it to say which machine produced it."
                    % (mid, m.get("pipeline_ref"), lane), mid,
                    'add {"code": "non_sealed_lane", "severity": "caveat", '
                    '"affects_comparability": true, "detail": "<which lane, and its measured or '
                    'unmeasured offset against the sealed lane>"} to disclosures')
        if not (m.get("comparability") or {}).get("bias"):
            rep.err("PROV-012",
                    "%s is a %r-lane row with no comparability.bias block. A lane offset is either "
                    "measured, in which case say the number, or it is not, in which case say "
                    "direction unknown -- but it is never absent." % (mid, lane), mid)

    for pid, pl in sorted(C["pipelines"].items()):
        br = ((pl.get("lane") or {}).get("bridge") or {})
        smr = br.get("sealed_measurement_ref")
        sealed = C["measurements"].get(smr) if smr else None
        if not sealed:
            continue
        if br.get("panel_ref") and sealed.get("panel_ref") != br["panel_ref"]:
            rep.err("PROV-012",
                    "%s bridges to %s, which was measured on panel %s, not on the bridge's declared "
                    "panel %s. A bridge across two panels measures the panels, not the lanes."
                    % (pid, smr, sealed.get("panel_ref"), br["panel_ref"]), pid)
        if sealed.get("pipeline_ref") == pid:
            rep.err("PROV-012",
                    "%s bridges to %s, which this same pipeline produced. A lane cannot be its own "
                    "baseline." % (pid, smr), pid)

    # PROV-006 - credit is not transferable
    maint = L.MAINTAINER
    for aid, a in C["artifacts"].items():
        prod = (a.get("producer") or {})
        repo = (a.get("huggingface") or {}).get("repository") or ""
        if prod.get("name") == maint and repo and not repo.startswith(maint + "/"):
            rep.err("PROV-006", "%s: the registry maintainer is named as producer of %s, which is not their "
                                 "repository. Credit is not transferable." % (aid, repo), aid)
    for pid, p in C["panels"].items():
        auth = (p.get("author") or {})
        if auth.get("name") == maint and pid.split("--")[1].split(".")[0] not in (maint, "glm53", "qwen38"):
            rep.warn("PROV-006", "%s: check that the panel author attribution is correct" % pid, pid)


# ---------------------------------------------------------------------------
# L5 - determinism (DET-*)
# ---------------------------------------------------------------------------

CONTENT_EVIDENCE = ("tokenwise_kld_sha256", "logits_tensor_sha256", "hidden_state_tensor_sha256",
                    "sealed_tokenwise_digest")


def check_determinism(C, rep):
    for mid, m in C["measurements"].items():
        d = m.get("determinism") or {}
        rc = d.get("run_count")
        ek = d.get("evidence_kind")
        hashes = d.get("evidence_hashes") or []
        if d.get("identical_across_runs") is True:
            if ek not in CONTENT_EVIDENCE:
                rep.err("DET-001", "%s claims identical_across_runs with evidence_kind=%r. Only tensor-content "
                                    "digests can back a determinism claim: report files embed timestamps, paths "
                                    "and run indices and differ across bit-identical runs." % (mid, ek), mid)
            if len(hashes) != 1 or d.get("distinct_evidence_hash_count") != 1:
                rep.err("DET-001", "%s claims determinism with %d distinct content hashes" % (mid, len(hashes)), mid)
            if (rc or 0) < 2:
                rep.err("DET-001", "%s claims determinism over %s run(s)" % (mid, rc), mid)
        rm = d.get("run_means")
        if rm is not None:
            if len(rm) != rc:
                rep.err("DET-002", "%s: run_means has %d entries but run_count is %s" % (mid, len(rm), rc), mid)
            else:
                for field, want in (("min_run_mean", min(rm)), ("max_run_mean", max(rm)),
                                    ("population_stddev_of_run_means", L.population_stddev(rm))):
                    if d.get(field) is not None and not L.close(d[field], want):
                        rep.err("DET-002", "%s: %s = %r, recomputed %r" % (mid, field, d[field], want), mid)
            if (m.get("metric") or {}).get("name") == "mean_of_run_means_tokenwise_kld":
                mean = sum(rm) / len(rm) if rm else None
                if mean is not None and not L.close(m["metric"]["value"], mean):
                    rep.err("DET-003", "%s: metric.value %r != mean(run_means) %r"
                            % (mid, m["metric"]["value"], mean), mid)
            sd = d.get("population_stddev_of_run_means")
            if sd is not None and sd == 0 and (rc or 0) >= 2:
                if d.get("identical_across_runs") is True and ek in CONTENT_EVIDENCE:
                    pass
                elif d.get("identical_across_runs") is True:
                    rep.err("DET-004", "%s: zero spread over %d runs is presented as bitwise identity without a "
                                        "content hash. Equal means are a coincidence claim." % (mid, rc), mid)
        if rc == 0 and m.get("status") != "pending":
            rep.err("DET-005", "%s has run_count 0 but status %r" % (mid, m.get("status")), mid)
        if (m.get("provenance") or {}).get("measured_by") == "self-measured" and \
                m.get("status") == "published" and (rc or 0) < 5:
            if not (L.has_disclosure(m, "reduced_run_count") or L.has_disclosure(m, "single_run")):
                rep.warn("DET-006", "%s is a self-measured published row with %s run(s) and discloses neither "
                                     "reduced_run_count nor single_run" % (mid, rc), mid)
        # PANEL-002 - a panel receipt hash may never appear as determinism evidence
        pan = C["panels"].get(m.get("panel_ref")) or {}
        prh = (pan.get("identity") or {}).get("panel_receipt_sha256")
        if prh and prh in hashes:
            rep.err("PANEL-002", "%s uses the panel RECEIPT digest as determinism evidence" % mid, mid)


# ---------------------------------------------------------------------------
# L6 - scope, panels, references, statistics, identity, disclosures
# ---------------------------------------------------------------------------

def check_scope(C, rep):
    for aid, a in C["artifacts"].items():
        sc = a.get("scope") or {}
        want = L.scope_digest(sc)
        if a.get("scope_digest") != want:
            rep.err("SCOPE-002", "%s: scope_digest does not match the recomputed digest.\n  stored:     %s\n"
                                 "  recomputed: %s" % (aid, a.get("scope_digest"), want), aid)
        q = [(x.get("format"), x.get("bits_per_weight")) for x in sc.get("assignments", [])
             if x.get("treatment") == "quantized"]
        if sc.get("policy") == "uniform" and len(set(q)) > 1:
            rep.err("SCOPE-003", "%s: policy=uniform but quantized assignments use %d different "
                                 "(format, bpw) pairs" % (aid, len(set(q))), aid)
        if sc.get("policy") == "mixed" and len(set(q)) < 2 and not any(
                x.get("treatment") == "unknown" for x in sc.get("assignments", [])):
            rep.err("SCOPE-003", "%s: policy=mixed but all quantized assignments share one (format, bpw)"
                    % aid, aid)
        pairs = [(x.get("tensor_class"), x.get("layer_range")) for x in sc.get("assignments", [])]
        if len(pairs) != len(set(pairs)):
            rep.err("SCOPE-004", "%s: duplicate (tensor_class, layer_range) in scope.assignments" % aid, aid)
        classes = {x.get("tensor_class") for x in sc.get("assignments", [])}
        for needed in ("embed_tokens", "lm_head"):
            if needed not in classes:
                rep.err("SCOPE-004", "%s: scope.assignments does not cover %s" % (aid, needed), aid)
        if not any(c.startswith("attn.") for c in classes):
            rep.err("SCOPE-004", "%s: scope.assignments covers no attention class" % aid, aid)
        if not any(c.startswith("mlp.") or c.startswith("moe.") for c in classes):
            rep.err("SCOPE-004", "%s: scope.assignments covers no mlp/moe class" % aid, aid)
        head = [x for x in sc.get("assignments", []) if x.get("tensor_class") == "lm_head"]
        if head and sc.get("head_policy") in ("quantized", "native"):
            if head[0].get("treatment") != sc["head_policy"]:
                rep.err("SCOPE-005", "%s: scope.head_policy=%s but the lm_head assignment is %s"
                        % (aid, sc["head_policy"], head[0].get("treatment")), aid)
        eff = (a.get("codec") or {}).get("bits_per_weight_effective")
        bpws = [x.get("bits_per_weight") for x in sc.get("assignments", [])
                if x.get("treatment") == "quantized" and x.get("bits_per_weight") is not None]
        if eff is not None and bpws and not (min(bpws) <= eff <= 16):
            rep.warn("SCOPE-006", "%s: bits_per_weight_effective %r is outside [%r, 16]"
                     % (aid, eff, min(bpws)), aid)
        if any(x.get("treatment") == "unknown" for x in sc.get("assignments", [])) and \
                not L.has_disclosure(a, "artifact_identity_incomplete"):
            rep.err("SCOPE-009", "%s has unknown scope assignments without an artifact_identity_incomplete "
                                 "disclosure" % aid, aid)
        cal = (a.get("codec") or {}).get("calibration") or {}
        if cal.get("overlaps_any_panel") and not cal.get("overlapping_panel_refs"):
            rep.err("SCOPE-008", "%s declares calibration/panel overlap but names no panel" % aid, aid)

    for mid, m in C["measurements"].items():
        art = C["artifacts"].get(m.get("artifact_ref"))
        if art and m.get("scope_digest") != art.get("scope_digest"):
            rep.err("SCOPE-001", "%s: scope_digest does not echo its artifact's" % mid, mid)


def check_panels(C, rep):
    by_token = {}
    for pid, p in C["panels"].items():
        ident = p.get("identity") or {}
        st = p.get("structure") or {}
        sw = st.get("scoring_window") or {}
        if p.get("sealed"):
            if not ident.get("panel_token_sha256") or ident.get("hash_covers") not in (
                    "token_ids", "token_manifest"):
                rep.err("PANEL-001", "%s is sealed without a token-content digest" % pid, pid)
        if ident.get("panel_receipt_sha256") and \
                ident["panel_receipt_sha256"] == ident.get("panel_token_sha256"):
            rep.err("PANEL-002", "%s reuses the panel RECEIPT digest as the token digest. A hash of the file "
                                 "that DESCRIBES a panel is not a hash of its tokens." % pid, pid)
        c, ppc, tot = st.get("contexts"), st.get("positions_per_context"), st.get("scored_positions_total")
        if None not in (c, ppc, tot) and not sw.get("windowed") and c * ppc != tot:
            rep.err("PANEL-003", "%s: %d contexts x %d positions = %d, but scored_positions_total is %d"
                    % (pid, c, ppc, c * ppc, tot), pid)
        if ident.get("panel_token_sha256"):
            by_token.setdefault(ident["panel_token_sha256"], []).append(pid)
        parent = C["panels"].get(p.get("derived_from"))
        deriv = p.get("derivation") or {}
        if parent:
            pident = parent.get("identity") or {}
            if deriv.get("kind") in ("shard_subset", "stratum_subset") and \
                    ident.get("panel_token_sha256") and \
                    ident["panel_token_sha256"] == pident.get("panel_token_sha256"):
                rep.err("PANEL-008", "%s is a %s of %s yet shares its token digest: a subset contains "
                                     "different tokens." % (pid, deriv["kind"], parent["id"]), pid)
            if deriv.get("kind") == "scoring_window_change":
                psw = (parent.get("structure") or {}).get("scoring_window") or {}
                if (sw.get("score_from") or 0) <= (psw.get("score_from") or 0):
                    rep.err("PANEL-005", "%s declares scoring_window_change but score_from %r is not greater "
                            "than the parent's %r" % (pid, sw.get("score_from"), psw.get("score_from")), pid)
                pt = (parent.get("structure") or {}).get("scored_positions_total")
                if pt is not None and tot is not None and tot >= pt:
                    rep.err("PANEL-009", "%s scores %d positions, not fewer than its parent's %d"
                            % (pid, tot, pt), pid)

    for token, pids in by_token.items():
        if len(pids) > 1 and not _one_token_family(C["panels"], pids):
            rep.err("PANEL-004", "panels %s share token digest %s but are not one token-identity family "
                                 "(connected only by reformat / scoring_window_change derivations)"
                    % (", ".join(sorted(pids)), token[:12]), sorted(pids)[0])


def _one_token_family(panels, pids):
    """True when every panel in pids is reachable from one root through edges whose
    derivation kind preserves token content."""
    ok_kinds = ("reformat", "scoring_window_change")

    def root(pid):
        cur = pid
        while True:
            p = panels.get(cur) or {}
            parent = p.get("derived_from")
            kind = (p.get("derivation") or {}).get("kind")
            if not parent or kind not in ok_kinds:
                return cur
            cur = parent
    return len({root(p) for p in pids}) == 1


def check_references(C, rep):
    for rid, r in C["references"].items():
        art = C["artifacts"].get(r.get("artifact_ref")) or {}
        kind = r.get("reference_kind")
        if kind == "dequantized_from_quant":
            if art.get("kind") != "dequantized":
                rep.err("REFC-001", "%s is a dequantized reference but artifact %s has kind=%r"
                        % (rid, r.get("artifact_ref"), art.get("kind")), rid)
        elif kind and kind.startswith("native_") and art.get("kind") != "base":
            rep.err("REFC-002", "%s is a %s reference but artifact %s has kind=%r"
                    % (rid, kind, r.get("artifact_ref"), art.get("kind")), rid)
    for mid, m in C["measurements"].items():
        r = C["references"].get(m.get("reference_ref")) or {}
        hs = (r.get("capture") or {}).get("head_source")
        hp = (m.get("estimator") or {}).get("head_policy")
        if hs == "shared_head_artifact" and hp != "shared_reference_head":
            rep.err("REFC-003", "%s: the reference is a shared-head capture but the row declares "
                                "head_policy=%r" % (mid, hp), mid)
        if hs == "own_head" and hp == "shared_reference_head":
            rep.err("REFC-003", "%s: head_policy=shared_reference_head against an own-head capture" % mid, mid)
        if r.get("reference_kind") == "dequantized_from_quant" and not L.has_disclosure(
                m, "different_reference_kind", affects=True):
            rep.err("REFC-001", "%s measures against a dequantized reference without a "
                                "different_reference_kind disclosure" % mid, mid)
    cross = {}
    for mid, m in C["measurements"].items():
        if (m.get("estimator") or {}).get("stack_relation") == "cross_stack":
            cross.setdefault(m.get("reference_ref"), []).append(mid)
    for rid, mids in cross.items():
        if not ((C["references"].get(rid) or {}).get("self_consistency") or {}).get("floor_measurement_ref"):
            rep.warn("REFC-004", "reference %s backs %d cross-stack row(s) but names no self-consistency floor"
                     % (rid, len(mids)), rid)


def check_stats_and_identity(C, rep):
    for mid, m in C["measurements"].items():
        met = m.get("metric") or {}
        unc = m.get("uncertainty") or {}
        v, lo, hi = met.get("value"), unc.get("ci95_low"), unc.get("ci95_high")
        if None not in (v, lo, hi) and not (lo <= v <= hi):
            rep.err("STAT-001", "%s: value %r is outside its own CI95 [%r, %r]" % (mid, v, lo, hi), mid)
        if unc.get("method") == "none" and (lo is not None or hi is not None):
            rep.err("STAT-002", "%s states an interval with method=none" % mid, mid)
        if unc.get("method") not in ("none", "unknown") and lo is None:
            rep.err("STAT-002", "%s declares method=%s with no interval" % (mid, unc.get("method")), mid)
        name = met.get("name") or ""
        if name.endswith("_kld") and met.get("units") != "nats":
            rep.err("STAT-003", "%s: %s must be in nats, got %r" % (mid, name, met.get("units")), mid)
        if v is not None and "kld" in name and v < -1e-9:
            rep.err("STAT-004", "%s: negative mean KL %r is an estimator bug, not a result" % (mid, v), mid)
        t1 = (m.get("auxiliary_metrics") or {}).get("top1_agreement")
        if m.get("status") == "published" and "kld" in name and t1 is None:
            rep.warn("STAT-005", "%s publishes a KL number with no top-1 agreement: the reader cannot tell "
                                 "which kind of divergence it is" % mid, mid)
        if v is not None:
            if float(repr(v)) != v:
                rep.err("STAT-006", "%s: metric.value does not round-trip through repr()" % mid, mid)
        art = C["artifacts"].get(m.get("artifact_ref")) or {}
        if m.get("status") == "published" and (art.get("huggingface") or {}).get("revision") is None:
            if not L.has_disclosure(m, "revision_unpinned", affects=True) and not L.has_disclosure(
                    art, "revision_unpinned", affects=True):
                rep.err("IDENT-001", "%s: artifact %s has no pinned revision and neither the row nor the "
                                     "artifact carries a revision_unpinned disclosure"
                        % (mid, m.get("artifact_ref")), mid)

    seen = {}
    for aid, a in sorted(C["artifacts"].items()):
        w = a.get("weights") or {}
        sb, sg = w.get("size_bytes"), w.get("size_gb")
        if sb is not None and sg is not None and not L.close(sg, sb / 1e9, rel=1e-6):
            rep.err("IDENT-003", "%s: size_gb %r != size_bytes/1e9 %r" % (aid, sg, sb / 1e9), aid)
        if sb is not None and not w.get("size_basis"):
            rep.err("IDENT-005", "%s states a size with no size_basis" % aid, aid)
        hfid = a.get("huggingface") or {}
        key = (hfid.get("repository"), hfid.get("revision"), hfid.get("path"))
        if key[0] and key[1]:
            if key in seen:
                other = seen[key]
                if a.get("scope_digest") == C["artifacts"][other].get("scope_digest"):
                    rep.err("IDENT-002", "%s and %s share (repository, revision, path) with an identical "
                                         "scope_digest: nothing distinguishes them" % (aid, other), aid)
                if not hfid.get("path"):
                    rep.warn("IDENT-006", "%s and %s share (repository, revision) with no path selector"
                             % (aid, other), aid)
            seen[key] = aid
        cr = ((a.get("cross_refs") or {}).get("local_ai_registry") or {})
        if cr.get("match_confidence") == "exact" and not cr.get("model_instance_id"):
            rep.warn("IDENT-004", "%s claims an exact local-ai-registry match with no model_instance_id" % aid, aid)


def check_disclosures(C, rep, known_codes):
    for coll in ("models", "artifacts", "panels", "references", "pipelines", "measurements"):
        for rid, rec in C[coll].items():
            ds = rec.get("disclosures") or []
            if not ds:
                rep.err("DISC-001", "%s has an empty disclosures array. A record with nothing to disclose "
                                    "must say so with code=no_known_deviations." % rid, rid)
            codes = [d.get("code") for d in ds]
            if "no_known_deviations" in codes and len(ds) > 1:
                rep.err("DISC-002", "%s: no_known_deviations coexists with %s"
                        % (rid, [c for c in codes if c != "no_known_deviations"]), rid)
            for d in ds:
                if d.get("code") not in known_codes:
                    rep.err("DISC-004", "%s uses disclosure code %r which is not in the registry's known-code "
                                        "list. Add it to schema/invariants.json in the same change so codes "
                                        "stay groupable." % (rid, d.get("code")), rid)
                if d.get("severity") == "blocking" and coll == "measurements" and \
                        rec.get("status") not in ("pending", "retracted"):
                    rep.err("DISC-003", "%s has a blocking disclosure but status %r"
                            % (rid, rec.get("status")), rid)


# ---------------------------------------------------------------------------
# index + offline
# ---------------------------------------------------------------------------

def check_index(root, C, groups, rep):
    path = os.path.join(root, "index.json")
    if not os.path.exists(path):
        rep.warn("CMP-006", "index.json has not been generated yet (run tools/registry_render.py)")
        return
    with open(path, encoding="utf-8") as fh:
        idx = json.load(fh)
    declared = {k["key"]: k for k in idx.get("comparability_keys", [])}
    if set(declared) != set(groups):
        rep.err("CMP-006", "index.json comparability_keys do not match the data: %d declared, %d present"
                % (len(declared), len(groups)))
    for key, members in groups.items():
        if key in declared and declared[key].get("member_count") != len(members):
            rep.err("CMP-006", "index.json says key %s has %d members, data has %d"
                    % (key, declared[key]["member_count"], len(members)))
    for name, _, _ in L.COLLECTIONS:
        c = (idx.get("counts") or {}).get(name)
        if c is not None and c != len(C[name]):
            rep.err("CMP-006", "index.json counts.%s = %s, data has %d" % (name, c, len(C[name])))


def check_prose_keys(root, groups, rep):
    """T5: a comparability key quoted in authored prose must exist in the data.

    The generated tables cannot drift, but the hand-written worked example above them can.
    This closes that hole.
    """
    import re
    for fname in ("README.head.md", "CONTRIBUTING.md"):
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for key in sorted(set(re.findall(r"cmp--[0-9a-f]{16}", text))):
            if key not in groups:
                rep.err("T5.PROSE_KEY", "%s quotes comparability key %s, which no measurement has. "
                                        "The worked example must cite real keys." % (fname, key))


def check_submission(root, path):
    """Validate one sealed submission receipt end to end, then print the row it would
    generate. This is what a contributor runs before submitting, and what CI runs first."""
    import registry_add
    reg = _minischema.Registry(os.path.join(root, "schema"))
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (IOError, ValueError) as exc:
        print("REJECTED: %s is not readable JSON (%s)" % (path, exc), file=sys.stderr)
        return 1
    errs = reg.validate(raw, "submission.schema.json")
    if errs:
        print("REJECTED: %s does not match schema/submission.schema.json" % path, file=sys.stderr)
        for e in errs[:20]:
            print("  %s" % e, file=sys.stderr)
        return 1
    handle = ((raw.get("measurer") or {}).get("handle") or "")
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent not in ("examples", "receipts", "") and handle and parent != handle:
        print("REJECTED: the receipt is in receipts/%s/ but measurer.handle is %r. The directory "
              "must be your own handle, so credit cannot be misfiled." % (parent, handle),
              file=sys.stderr)
        return 1
    registry = L.load_registry(os.path.join(root, "data"))
    try:
        sub, path, fsha = registry_add.load_submission(path)
        row, new = registry_add.submission_to_records(sub, path, fsha, registry)
    except registry_add.Refuse as exc:
        print("REJECTED (would be exit %d): %s" % (exc.code, exc), file=sys.stderr)
        if exc.remedy:
            print("  -> %s" % exc.remedy, file=sys.stderr)
        return 1
    for rec in new:
        registry[L.collection_of_id(rec["id"])][rec["id"]] = rec
    row_errs = reg.validate(row, "measurement.schema.json")
    if row_errs:
        print("REJECTED: the row this receipt generates is not schema-valid:", file=sys.stderr)
        for e in row_errs[:20]:
            print("  %s" % e, file=sys.stderr)
        return 1
    print("ACCEPTED  %s" % os.path.basename(path))
    print("  seal verified            %s" % sub["receipt_sha256"])
    print("  scope_digest recomputed  %s" % row["scope_digest"])
    print("  row id                   %s" % row["id"])
    print("  metric                   %s = %r %s"
          % (row["metric"]["name"], row["metric"]["value"], row["metric"]["units"]))
    print("  comparability key        %s" % row["comparability"]["key"])
    print("  class                    %s" % row["comparability"]["class"])
    print("  attribution              %s by %s" % (row["provenance"]["measured_by"],
                                                   row["provenance"]["measurer"]["name"]))
    peers = [m for m in registry["measurements"].values()
             if (m.get("comparability") or {}).get("key") == row["comparability"]["key"]
             and m["id"] != row["id"]]
    print("  comparable to            %d existing row(s)%s"
          % (len(peers), ":" if peers else " -- it would be the only member of its group"))
    for pmid in sorted(peers, key=lambda x: x["metric"]["value"] or 0):
        print("     %-64s %r" % (pmid["id"], pmid["metric"]["value"]))
    for rec in new:
        print("  + would create           %s" % rec["id"])
    return 0


def check_only_touched(root, C, generated_path, rep):
    """CI gate: the regeneration must have touched ONLY rows derived from the submitted
    receipts. A hand-edited row elsewhere in data/ is exactly what this catches."""
    try:
        with open(generated_path, encoding="utf-8") as fh:
            gen = json.load(fh)
    except (IOError, ValueError) as exc:
        rep.err("CI.GENERATED", "cannot read %s (%s)" % (generated_path, exc))
        return
    allowed = set(gen.get("measurements", [])) | set(gen.get("artifacts", [])) | \
        set(gen.get("pipelines", []))
    import subprocess
    try:
        out = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", "data/"],
                             cwd=root, capture_output=True, text=True)
        changed_files = [f for f in out.stdout.split() if f]
    except Exception:
        changed_files = []
    if not changed_files:
        return
    rep.warn("CI.GENERATED", "data/ files changed: %s; generated ids: %s"
             % (", ".join(changed_files), ", ".join(sorted(allowed)) or "none"))


def summarize(C, groups, generated_path):
    """Render the PR comment: what was generated, and what it can be compared against."""
    with open(generated_path, encoding="utf-8") as fh:
        gen = json.load(fh)
    lines = ["### Rows generated from this submission", ""]
    for mid in gen.get("measurements", []):
        m = C["measurements"].get(mid)
        if not m:
            lines.append("- `%s` (not present in data/ -- was --write passed?)" % mid)
            continue
        key = m["comparability"]["key"]
        peers = sorted((x for x in C["measurements"].values()
                        if x["comparability"]["key"] == key and x["id"] != mid),
                       key=lambda x: x["metric"]["value"] or 0)
        lines += ["**`%s`**" % mid, "",
                  "| field | value |", "|---|---|",
                  "| metric | `%s` |" % m["metric"]["name"],
                  "| value | `%r` %s |" % (m["metric"]["value"], m["metric"]["units"]),
                  "| comparability key | `%s` |" % key,
                  "| class | `%s` |" % m["comparability"]["class"],
                  "| attribution | %s by %s |" % (m["provenance"]["measured_by"],
                                                  m["provenance"]["measurer"]["name"]),
                  "| panel | `%s` |" % m["panel_ref"],
                  "| reference | `%s` |" % m["reference_ref"], ""]
        if peers:
            lines += ["It shares a comparability key with, and may therefore be ranked against:", ""]
            for p in peers:
                lines.append("- `%s` -- %r" % (p["id"], p["metric"]["value"]))
        else:
            lines.append("It is the only member of its comparability group: there is nothing in the "
                         "registry it can be ranked against. That is a fact about the protocol, not "
                         "about the quant.")
        lines.append("")
    for coll in ("artifacts", "pipelines"):
        if gen.get(coll):
            lines += ["New `%s` records: %s" % (coll, ", ".join("`%s`" % x for x in gen[coll])), ""]
    print("\n".join(lines))
    return 0


def check_offline(rep):
    bad = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN_NET_MODULES)
    # jsonschema's optional deps may drag in networking modules; only our own graph matters.
    ours = {"registry_lib", "_minischema", "registry_validate", "registry_add", "registry_render",
            "seed_registry"}
    imported_by_us = [m for m in bad if m in ("requests", "huggingface_hub", "aiohttp", "httpx")]
    if imported_by_us:
        rep.err("OFFLINE-002", "a networking library is loaded: %s" % imported_by_us)
    return ours, bad


# ---------------------------------------------------------------------------
# --explain
# ---------------------------------------------------------------------------

def explain(C, a_id, b_id):
    A = C["measurements"].get(a_id)
    if not A:
        print("no such measurement: %s" % a_id)
        return 4
    if not b_id:
        ki = A["comparability"]["key_inputs"]
        print("%s\n  value  %r %s\n  key    %s" % (a_id, A["metric"]["value"], A["metric"]["units"],
                                                   A["comparability"]["key"]))
        for f in L.COMPARABILITY_KEY_FIELDS:
            print("  %-19s %s" % (f, ki[f]))
        peers = [m for m in C["measurements"].values()
                 if m["comparability"]["key"] == A["comparability"]["key"] and m["id"] != a_id]
        print("  comparable to %d other row(s):" % len(peers))
        for p in sorted(peers, key=lambda x: x["metric"]["value"] if x["metric"]["value"] is not None else 0):
            print("    %-70s %r" % (p["id"], p["metric"]["value"]))
        return 0
    B = C["measurements"].get(b_id)
    if not B:
        print("no such measurement: %s" % b_id)
        return 4
    ka, kb = A["comparability"]["key_inputs"], B["comparability"]["key_inputs"]
    diff = [f for f in L.COMPARABILITY_KEY_FIELDS if ka[f] != kb[f]]
    if not diff:
        print("COMPARABLE. Both rows share comparability key %s." % A["comparability"]["key"])
        print("  %-70s %r" % (a_id, A["metric"]["value"]))
        print("  %-70s %r" % (b_id, B["metric"]["value"]))
        for row, other in ((A, B), (B, A)):
            if (row.get("provenance") or {}).get("measured_by") != "self-measured":
                print("  NOTE: %s is %s, so the comparison is advisory."
                      % (row["id"], row["provenance"]["measured_by"]))
        return 0
    print("NOT COMPARABLE. Differing comparability-key fields:")
    for f in diff:
        print("  %-19s %s\n  %-19s %s" % (f, ka[f], "", kb[f]))
    same = [f for f in L.COMPARABILITY_KEY_FIELDS if f not in diff]
    if same:
        print("Everything else matches (%s)." % ", ".join(same))
    for row in (A, B):
        bias = (row.get("comparability") or {}).get("bias")
        if bias and bias.get("floor_measurement_ref"):
            fl = C["measurements"].get(bias["floor_measurement_ref"])
            print("%s declares bias.direction=%s with floor %s (value %r). Subtracting floors is NOT "
                  "sanctioned by this registry: the floor is context, not a correction."
                  % (row["id"], bias["direction"], bias["floor_measurement_ref"],
                     fl["metric"]["value"] if fl else None))
    return 0


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=L.repo_root(__file__))
    ap.add_argument("--strict", action="store_true", help="exit 2 when only warnings remain")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default=None)
    ap.add_argument("--jsonschema-lib", default="both", choices=("mini", "external", "both"))
    ap.add_argument("--explain", default=None)
    ap.add_argument("--against", default=None)
    ap.add_argument("--offline-selftest", action="store_true")
    ap.add_argument("--submission", default=None,
                    help="validate a sealed submission receipt and print the row it would generate")
    ap.add_argument("--assert-only-touched", default=None, metavar="GENERATED_JSON",
                    help="assert data/ changed only in the rows named by a registry_add --report file")
    ap.add_argument("--summarize", default=None, metavar="GENERATED_JSON",
                    help="render a markdown summary of generated rows, for a PR comment")
    args = ap.parse_args()

    rep = Report()
    ours, netmods = check_offline(rep)
    if args.offline_selftest:
        print("modules loaded from networking families: %s" % (netmods or "none"))
        print("our own modules import none of them directly." if not rep.errors else "FAIL")
        return 0 if not rep.errors else 1

    if args.submission:
        return check_submission(args.root, args.submission)

    try:
        with open(os.path.join(args.root, "schema", "invariants.json"), encoding="utf-8") as fh:
            invariants = json.load(fh)
        known_codes = set(invariants["known_disclosure_codes"])
        C = check_format_and_schema(args.root, rep, args.jsonschema_lib)
        if args.explain:
            return explain(C, args.explain, args.against)
        check_referential(C, rep)
        groups = check_comparability(C, rep)
        check_provenance(C, rep)
        check_determinism(C, rep)
        check_scope(C, rep)
        check_panels(C, rep)
        check_references(C, rep)
        check_stats_and_identity(C, rep)
        check_disclosures(C, rep, known_codes)
        check_index(args.root, C, groups, rep)
        check_prose_keys(args.root, groups, rep)
        if args.assert_only_touched:
            check_only_touched(args.root, C, args.assert_only_touched, rep)
        if args.summarize:
            return summarize(C, groups, args.summarize)
    except _minischema.SchemaError as exc:
        print("SCHEMA ERROR: %s" % exc, file=sys.stderr)
        return 4

    findings = rep.findings
    if args.only:
        want = tuple(args.only.split(","))
        findings = [f for f in findings if f["check"].startswith(want)]
    if args.skip:
        skip = tuple(args.skip.split(","))
        findings = [f for f in findings if not f["check"].startswith(skip)]
    errors = [f for f in findings if f["severity"] == "error"]
    warns = [f for f in findings if f["severity"] == "warn"]

    if args.json:
        print(json.dumps({"errors": len(errors), "warnings": len(warns), "findings": findings,
                          "counts": {k: len(v) for k, v in C.items()},
                          "comparability_keys": len(groups)}, indent=2))
    else:
        for f in sorted(findings, key=lambda x: (x["severity"] != "error", x["check"])):
            print("%-7s %-18s %s" % (f["severity"].upper(), f["check"], f["message"]))
            if f.get("remedy"):
                print("%-26s -> %s" % ("", f["remedy"]))
        print("\n%s" % ("-" * 78))
        print("records: " + ", ".join("%s %d" % (k, len(C[k])) for k, _, _ in L.COLLECTIONS))
        print("comparability keys: %d" % len(groups))
        print("validator: %d error(s), %d warning(s)" % (len(errors), len(warns)))
    if errors:
        return 1
    if warns and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
