# Everything here runs OFFLINE on a stock interpreter with no pip install.
# `make check` is what CI runs and what a contributor should run before submitting.

PY ?= python3
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: check validate render render-check selftest joint reseed reseed-check explain clean

check: validate render-check selftest joint
	@echo
	@echo "OK: schema + invariants clean, README tables match the data, self-tests pass."

validate:
	$(PY) tools/registry_validate.py --strict --jsonschema-lib mini || \
	  ( [ $$? -eq 2 ] && echo "(warnings only)" )

# Cross-check the vendored validator against the real library when it is importable.
validate-both:
	$(PY) tools/registry_validate.py --jsonschema-lib both

render:
	$(PY) tools/registry_render.py

render-check:
	$(PY) tools/registry_render.py --check

selftest:
	$(PY) tools/registry_selftest.py

# Joint-standard invariants (JOINT-001..008) that JSON Schema cannot express:
# the quadrature arithmetic, the per-domain partition, and the rule that a row's
# value must be the equal-weight mean of exactly the windows its scope names.
joint:
	$(PY) tools/registry_joint_check.py

# The seeded rows are a function of their receipts; this proves they still are.
reseed:
	$(PY) tools/seed_registry.py
	$(PY) tools/registry_render.py

reseed-check:
	$(PY) tools/seed_registry.py --check

# make explain A=measurement--... B=measurement--...
explain:
	$(PY) tools/registry_validate.py --explain "$(A)" $(if $(B),--against "$(B)")

clean:
	rm -rf tools/__pycache__
