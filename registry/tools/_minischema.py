"""A small, dependency-free JSON Schema draft 2020-12 validator.

Scope: exactly the keyword subset the quant-fidelity-registry schemas use --
$ref (local "#/$defs/..." and sibling-file "name.schema.json#/$defs/..."),
type, enum, const, required, properties, additionalProperties, items, prefixItems,
contains, minItems, maxItems, uniqueItems, minLength, maxLength, pattern,
minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf,
allOf, anyOf, oneOf, not, if/then/else, propertyNames, dependentRequired.
`format` is treated as an annotation, exactly as draft 2020-12 specifies by default.

Why this exists: the registry must validate on a stock interpreter with no
network and no pip. macOS ships Python 3.9 without `jsonschema` (verified: the
system interpreter here is 3.9.6, and `make check` runs clean under it).

It is also what keeps OFFLINE-002 honest. registry_validate asserts that none of
FORBIDDEN_NET_MODULES is loaded, and registry_selftest section E re-checks it;
the real `jsonschema`'s optional dependency closure is exactly what that
assertion is designed to exclude -- registry_validate:check_offline already has
to carve out an exception for it.

Drift protection is DIFFERENTIAL, not aspirational: `_external_validator` builds
a real jsonschema.Draft202012Validator over the same schema set and
`--jsonschema-lib both` runs both and compares. Note the split between the tool
and the gate -- the flag DEFAULTS to `both`, but `make check` -> `validate`
passes `--jsonschema-lib mini` explicitly, so the cross-check is not part of the
default gate. Run it deliberately, under an interpreter that has the library:

    make validate-both          # or: python3 tools/registry_validate.py --jsonschema-lib both

Last run 2026-08-31 against jsonschema 4.26.0: 0 errors over all 157 records.
It degrades gracefully (returns None) where the library is absent, so the command
is safe on a stock interpreter -- it simply has nothing to compare against.

Not a general-purpose validator. It raises on any keyword it does not implement
rather than ignoring it, so a schema that grows a new keyword fails loudly here
instead of silently validating nothing.
"""

import json
import os
import re

SUPPORTED = {
    "$schema", "$id", "$defs", "$ref", "$comment", "title", "description", "default", "examples",
    "deprecated", "readOnly", "writeOnly", "format",
    "type", "enum", "const", "required", "properties", "patternProperties", "additionalProperties",
    "propertyNames", "dependentRequired", "items", "prefixItems", "contains", "minContains",
    "maxContains", "minItems", "maxItems", "uniqueItems", "minLength", "maxLength", "pattern",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minProperties", "maxProperties",
    "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
}

_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "null": type(None),
}


class SchemaError(Exception):
    pass


class Error(object):
    __slots__ = ("path", "message", "schema_path")

    def __init__(self, path, message, schema_path=""):
        self.path = path
        self.message = message
        self.schema_path = schema_path

    def __str__(self):
        return "%s: %s" % (self.path or "<root>", self.message)


class Registry(object):
    """Loads every *.schema.json in a directory and resolves refs between them."""

    def __init__(self, schema_dir):
        self.dir = schema_dir
        self.docs = {}
        for name in sorted(os.listdir(schema_dir)):
            if name.endswith(".schema.json"):
                with open(os.path.join(schema_dir, name), "r", encoding="utf-8") as fh:
                    self.docs[name] = json.load(fh)
        for name, doc in self.docs.items():
            _assert_supported(doc, name)

    def resolve(self, ref, current_doc_name):
        if ref.startswith("#"):
            doc_name, pointer = current_doc_name, ref[1:]
        else:
            doc_name, _, frag = ref.partition("#")
            pointer = frag
        if doc_name not in self.docs:
            raise SchemaError("unresolvable $ref %r from %s" % (ref, current_doc_name))
        node = self.docs[doc_name]
        for part in [p for p in pointer.split("/") if p]:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or part not in node:
                raise SchemaError("unresolvable pointer %r in %s" % (pointer, doc_name))
            node = node[part]
        return node, doc_name

    def validate(self, instance, schema_name):
        errors = []
        _validate(instance, self.docs[schema_name], self, schema_name, "", errors)
        return errors


# REG-23. Keywords whose values are SCHEMAS. Everything else is a leaf: recursing into a
# DATA position made `{"const": {"a": 1}}` -- spec-legal, a is an instance property --
# raise "unsupported keyword 'a'", which registry_validate turns into exit 4 for the whole
# run. `default`, `examples` and object-valued `enum` members had the same problem, and
# they survived only by accident when their keys happened to be schema keywords. A
# whitelist of applicators is closed under future additions to SUPPORTED; a blacklist of
# data keywords has to be re-audited every time one is added.
_MAP_OF_SCHEMAS = ("properties", "$defs", "patternProperties")
_SCHEMA_VALUED = ("additionalProperties", "propertyNames", "items", "contains", "not",
                  "if", "then", "else", "prefixItems", "allOf", "anyOf", "oneOf")


def _assert_supported(node, where, path="#"):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _MAP_OF_SCHEMAS:
                if not isinstance(v, dict):
                    raise SchemaError("%s: %r must be an object of schemas at %s"
                                      % (where, k, path))
                for kk, vv in v.items():
                    _assert_supported(vv, where, path + "/" + k + "/" + kk)
                continue
            if k not in SUPPORTED:
                raise SchemaError("%s: unsupported keyword %r at %s" % (where, k, path))
            # dependentRequired's values are lists of property NAMES, not schemas.
            if k in _SCHEMA_VALUED:
                _assert_supported(v, where, path + "/" + k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _assert_supported(v, where, "%s/%d" % (path, i))


def _is_type(value, tname):
    if tname == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tname == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer())
    if tname == "boolean":
        return isinstance(value, bool)
    if tname == "object":
        return isinstance(value, dict)
    if tname == "array":
        return isinstance(value, list)
    if tname == "string":
        return isinstance(value, str)
    if tname == "null":
        return value is None
    raise SchemaError("unknown type %r" % tname)


def _canon(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def _ok(instance, schema, reg, doc, ):
    errs = []
    _validate(instance, schema, reg, doc, "", errs)
    return not errs


def _validate(inst, schema, reg, doc, path, errors):
    if schema is True or schema == {}:
        return
    if schema is False:
        errors.append(Error(path, "schema is false: no value is valid here"))
        return
    if not isinstance(schema, dict):
        raise SchemaError("schema must be an object or boolean, got %r" % type(schema))

    if "$ref" in schema:
        target, target_doc = reg.resolve(schema["$ref"], doc)
        _validate(inst, target, reg, target_doc, path, errors)
        # 2020-12: $ref no longer suppresses sibling keywords.

    if "type" in schema:
        types = schema["type"]
        types = types if isinstance(types, list) else [types]
        if not any(_is_type(inst, t) for t in types):
            errors.append(Error(path, "expected type %s, got %s" % ("/".join(types), _tname(inst))))
            return

    if "enum" in schema and not any(_canon(inst) == _canon(e) for e in schema["enum"]):
        errors.append(Error(path, "value %r is not one of %s" % (inst, schema["enum"])))
    if "const" in schema and _canon(inst) != _canon(schema["const"]):
        errors.append(Error(path, "value %r != const %r" % (inst, schema["const"])))

    if isinstance(inst, str):
        if "minLength" in schema and len(inst) < schema["minLength"]:
            errors.append(Error(path, "shorter than minLength %d" % schema["minLength"]))
        if "maxLength" in schema and len(inst) > schema["maxLength"]:
            errors.append(Error(path, "longer than maxLength %d" % schema["maxLength"]))
        if "pattern" in schema and re.search(schema["pattern"], inst) is None:
            errors.append(Error(path, "%r does not match pattern %r" % (inst[:80], schema["pattern"])))

    if isinstance(inst, (int, float)) and not isinstance(inst, bool):
        if "minimum" in schema and inst < schema["minimum"]:
            errors.append(Error(path, "%r < minimum %r" % (inst, schema["minimum"])))
        if "maximum" in schema and inst > schema["maximum"]:
            errors.append(Error(path, "%r > maximum %r" % (inst, schema["maximum"])))
        if "exclusiveMinimum" in schema and inst <= schema["exclusiveMinimum"]:
            errors.append(Error(path, "%r <= exclusiveMinimum" % inst))
        if "exclusiveMaximum" in schema and inst >= schema["exclusiveMaximum"]:
            errors.append(Error(path, "%r >= exclusiveMaximum" % inst))
        if "multipleOf" in schema and schema["multipleOf"] and (inst % schema["multipleOf"]) != 0:
            errors.append(Error(path, "%r is not a multiple of %r" % (inst, schema["multipleOf"])))

    if isinstance(inst, dict):
        for key in schema.get("required", []):
            if key not in inst:
                errors.append(Error(path, "missing required property %r" % key))
        if "minProperties" in schema and len(inst) < schema["minProperties"]:
            errors.append(Error(path, "fewer than minProperties"))
        if "maxProperties" in schema and len(inst) > schema["maxProperties"]:
            errors.append(Error(path, "more than maxProperties"))
        props = schema.get("properties", {})
        pat = schema.get("patternProperties", {})
        for key, value in inst.items():
            kp = "%s/%s" % (path, key)
            matched = False
            if key in props:
                matched = True
                _validate(value, props[key], reg, doc, kp, errors)
            for prx, sub in pat.items():
                if re.search(prx, key):
                    matched = True
                    _validate(value, sub, reg, doc, kp, errors)
            if not matched and "additionalProperties" in schema:
                ap = schema["additionalProperties"]
                if ap is False:
                    errors.append(Error(path, "additional property %r is not allowed" % key))
                else:
                    _validate(value, ap, reg, doc, kp, errors)
            if "propertyNames" in schema:
                _validate(key, schema["propertyNames"], reg, doc, kp, errors)
        for key, needed in schema.get("dependentRequired", {}).items():
            if key in inst:
                for n in needed:
                    if n not in inst:
                        errors.append(Error(path, "property %r requires %r" % (key, n)))

    if isinstance(inst, list):
        if "minItems" in schema and len(inst) < schema["minItems"]:
            errors.append(Error(path, "fewer than minItems %d" % schema["minItems"]))
        if "maxItems" in schema and len(inst) > schema["maxItems"]:
            errors.append(Error(path, "more than maxItems %d" % schema["maxItems"]))
        if schema.get("uniqueItems") and len({_canon(i) for i in inst}) != len(inst):
            errors.append(Error(path, "items are not unique"))
        start = 0
        for i, sub in enumerate(schema.get("prefixItems", [])):
            if i < len(inst):
                _validate(inst[i], sub, reg, doc, "%s/%d" % (path, i), errors)
            start = i + 1
        if "items" in schema:
            for i in range(start, len(inst)):
                _validate(inst[i], schema["items"], reg, doc, "%s/%d" % (path, i), errors)
        if "contains" in schema:
            n = sum(1 for i in inst if _ok(i, schema["contains"], reg, doc))
            if n < schema.get("minContains", 1):
                errors.append(Error(path, "no item matches 'contains'"))
            if "maxContains" in schema and n > schema["maxContains"]:
                errors.append(Error(path, "too many items match 'contains'"))

    for sub in schema.get("allOf", []):
        _validate(inst, sub, reg, doc, path, errors)
    if "anyOf" in schema and not any(_ok(inst, s, reg, doc) for s in schema["anyOf"]):
        errors.append(Error(path, "matches none of the anyOf branches"))
    if "oneOf" in schema:
        n = sum(1 for s in schema["oneOf"] if _ok(inst, s, reg, doc))
        if n != 1:
            errors.append(Error(path, "matches %d oneOf branches, expected exactly 1" % n))
    if "not" in schema and _ok(inst, schema["not"], reg, doc):
        errors.append(Error(path, "matches a 'not' schema that must not match"))
    if "if" in schema:
        if _ok(inst, schema["if"], reg, doc):
            if "then" in schema:
                _validate(inst, schema["then"], reg, doc, path, errors)
        elif "else" in schema:
            _validate(inst, schema["else"], reg, doc, path, errors)


def _tname(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, str):
        return "string"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__
