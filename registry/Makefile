# Everything here runs OFFLINE. `validate`, `render-check`, `selftest` and `joint` run
# on a stock interpreter with no pip install; `reseed` and `reseed-check` additionally
# need numpy, because the CI endpoints they re-derive come from a PCG64 resample
# stream that must match the joint standard's reference implementation bit-for-bit
# (the stdlib fallback draws a different stream from the same seed -- STAT-03).
# `make check` is what CI runs and what a contributor should run before submitting.

PY ?= python3
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: check check-release validate validate-release render render-check selftest joint \
        reseed reseed-check explain clean stat-selftest coverage

check: validate render-check selftest joint
	@echo
	@echo "OK: schema + invariants clean, README tables match the data, self-tests pass."
	@echo "    (validator WARNINGS do not block this target -- the banner above states"
	@echo "     the count. 'make check-release' is the gate that fails on them.)"

# Release gate: identical to `check` except that validator warnings FAIL it.
# 78 warnings converted into a green exit is how evidence gaps stay invisible;
# a release must look at zero.
check-release: validate-release render-check selftest joint
	@echo
	@echo "OK (release): schema + invariants clean with ZERO validator warnings."

# Dev gate: warnings do not block, but they are counted OUT LOUD, never
# laundered into silence. The validator's own exit code 2 means "warnings only".
validate:
	@out=$$(mktemp); \
	$(PY) tools/registry_validate.py --strict --jsonschema-lib mini >$$out 2>&1; rc=$$?; \
	cat $$out; \
	warnings=$$(sed -n 's/.*validator: [0-9][0-9]* error(s), \([0-9][0-9]*\) warning(s).*/\1/p' $$out | tail -1); \
	rm -f $$out; \
	if [ "$$rc" = 2 ]; then \
	  echo ""; \
	  echo "================================================================================"; \
	  echo "  VALIDATOR WARNINGS: $${warnings:-unknown count}."; \
	  echo "  'make check' stays green on warnings for dev flow ONLY; they are not noise"; \
	  echo "  and not resolved. 'make check-release' FAILS on them -- do not cut a release"; \
	  echo "  while this count is nonzero without disposition."; \
	  echo "================================================================================"; \
	elif [ "$$rc" != 0 ]; then exit "$$rc"; fi

validate-release:
	@$(PY) tools/registry_validate.py --strict --jsonschema-lib mini; rc=$$?; \
	if [ "$$rc" = 2 ]; then \
	  echo ""; \
	  echo "check-release: FAILED -- the validator reports warnings (exit 2) and the"; \
	  echo "release gate does not convert warnings into success. Disposition each one"; \
	  echo "or run plain 'make check' for dev flow."; \
	  exit 2; \
	elif [ "$$rc" != 0 ]; then exit "$$rc"; fi

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

# The per-domain interval's own arithmetic: measured coverage, per-stratum seeds,
# and the proof that the PRE-reseed endpoints still regenerate so the 2026-08-30
# delta stays attributable. Needs numpy, so it is NOT part of `check` -- `check`
# is the command a contributor runs on a stock interpreter with nothing installed.
# bin/selftest_all.sh runs it.
stat-selftest:
	$(PY) tools/selftest_stat01_reseed.py

# Regenerate the measured-coverage record the by_domain blocks quote. Minutes, not
# seconds: 54 cells x 4000 replications x up to 20000 resamples. The output is
# committed, so `make check` never pays for it and a reader can still reproduce it.
coverage:
	$(PY) tools/coverage_sim.py --reps 4000 \
	  --out protocol/coverage/domain-interval-coverage.v1.json

# make explain A=measurement--... B=measurement--...
explain:
	$(PY) tools/registry_validate.py --explain "$(A)" $(if $(B),--against "$(B)")

clean:
	rm -rf tools/__pycache__
