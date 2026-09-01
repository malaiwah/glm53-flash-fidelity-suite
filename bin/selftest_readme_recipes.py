#!/usr/bin/env python3
"""Selftest: every fenced command in the README's recipe sections is executable
as written — it parses against the real CLI it names, and the prose makes the
plan-only/cannot-fetch limits explicit.

The defect this guards against (contributor-experience review, 2026-08-31):
the local recipe was **not executable as written**. It omitted `--execute`
(measure-local is plan-only by default and exits 3 on a clean plan), and said
nothing about measure-local being unable to fetch the artifact, the teacher or
the pipeline — so a stranger pasted it, got a refusal, and could not tell
whether they had broken something. Docs had also drifted from argparse before
(the 2026-08 flag reconciliation found five documented flags that did not
exist), so this test PROBES the CLI rather than trusting the prose, the same
rule the engine pinning uses.

Rungs:
  R1  every `bin/<tool>` command in a ```bash fence in the recipe sections
      parses against that tool's real argparse parser (placeholders
      substituted; unknown flags or missing required args fail the rung).
  R2  wrapper commands with no importable parser (registry-submit) exist and
      are executable, and their delegate script exists.
  R3  Recipe 2 states the plan-only default and shows `--execute`.
  R4  Recipe 2 states that measure-local downloads nothing and names all
      three local inputs (--artifact-path, --teacher-tree, --pipeline-root).
  R5  Recipe 2 names the local lanes' surface limit (packed / native-bf16)
      or links to the support matrix.

Stock python3, stdlib only, offline, $0.00 — parsing never contacts anything.
"""

import contextlib
import io
import re
import shlex
import sys
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parent.parent
BIN = SUITE_ROOT / "bin"
README = SUITE_ROOT / "README.md"
sys.path.insert(0, str(BIN))

failures = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          ("  (%s)" % detail) if (detail and not ok) else ""))
    if not ok:
        failures.append(name)


# The recipe sections under test: from the one-command headline through the
# submit recipe, plus the registry browser section.
SECTION_START = "## Measure a quant from an HF link"
SECTION_END = "### Which cloud?"

# Placeholders a recipe legitimately uses; substituted before parsing.
PLACEHOLDERS = {
    "<hf-repo>": "example-org/example-quant",
    "<hf-dataset>": "example-org/example-panel",
    "<hf-url-or-repo>": "example-org/example-quant",
    "<out>": "/tmp/example-out",
    "<receipt.json>": "/tmp/example-out/receipts/measurement-receipt.json",
    "<panel>": "example-org/example-panel",
    "<rev>": "0" * 40,
    "<your-hf-handle>": "example-handle",
    "<token>": "EXAMPLE",
    "...": "PLACEHOLDER",
}

SHELL_PLACEHOLDERS = {
    "$RUNPOD_KEY_FILE": "/tmp/example-runpod-key",
    "$FIDELITY_STATE": "/tmp/example-fidelity-state",
    "$CAMPAIGN_LEDGER": "/tmp/example-fidelity-state/campaign.json",
    "$CAMPAIGN_CEILING_USD": "100",
    "$CAMPAIGN_RESERVE_USD": "10",
    "$CAMPAIGN_REAPER_MARGIN_USD": "2",
    "$DRILL_CAP_USD": "5",
    "$ATTEMPT_CAP_USD": "20",
    "$ROOT_DATASET_ID": "example-root",
    "$ROOT_DATASET_REPOSITORY": "example-org/example-root",
    "$ROOT_DATASET_NAME": "Example Root",
    "$ROOT_ATTEMPT_CAP_USD": "20",
    "$ROOT_MAX_RUNTIME": "12h",
    "$HOME": "/tmp/example-home",
}

# argv[0] -> module with build_parser(). Probed, never guessed.
PARSER_MODULES = {
    "bin/measure": "measure_one",
    "bin/measure-cloud": "measure_cloud",
    "bin/measure-local": "measure_local",
    "bin/registry-view": "registry_view",
}

# Wrappers whose contract is checked structurally (no importable parser).
WRAPPERS = {
    "bin/registry-submit": "registry/tools/registry_validate.py",
    "bin/fidelity-doctor": "bin/fidelity-doctor",
    "bin/fixture": "bin/fixture_fetch.py",
    "bin/fidelity-bench": "bin/fidelity/bench.py",
}


def recipe_text():
    text = README.read_text(encoding="utf-8")
    start = text.index(SECTION_START)
    end = text.index(SECTION_END, start)
    return text[start:end]


def bash_blocks(text):
    return re.findall(r"```bash\n(.*?)```", text, re.S)


def commands(block):
    """Join continuations, drop comments/exports/output lines, keep bin/*."""
    joined = re.sub(r"\\\n\s*", " ", block)
    out = []
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("export "):
            continue
        if line.startswith("$ "):
            line = line[2:]
        if line.startswith("./"):
            line = line[2:]
        # Trailing inline comments (recipes annotate commands; no recipe
        # quotes a literal '#').
        line = re.split(r"\s+#", line)[0].strip()
        if line.startswith("bin/"):
            out.append(line)
    return out


def substitute(cmd):
    for key in sorted(SHELL_PLACEHOLDERS, key=len, reverse=True):
        cmd = cmd.replace(key, SHELL_PLACEHOLDERS[key])
    for key, value in PLACEHOLDERS.items():
        cmd = cmd.replace(key, value)
    return cmd


def try_parse(module_name, argv):
    import importlib
    mod = importlib.import_module(module_name)
    parser = mod.build_parser()
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(argv)
        return None
    except SystemExit as exc:
        return "argparse exit %s" % exc.code
    except Exception as exc:                                  # noqa: BLE001
        return "parser raised %s: %s" % (type(exc).__name__, exc)


def main():
    section = recipe_text()
    blocks = bash_blocks(section)
    # The third-party quickstart's fenced commands are recipes too — same
    # contract: executable as written, or explicitly $0/refusing.
    quickstart = SUITE_ROOT / "docs" / "THIRD-PARTY-QUICKSTART.md"
    if quickstart.is_file():
        blocks += bash_blocks(quickstart.read_text(encoding="utf-8"))
    cmds = [substitute(c) for b in blocks for c in commands(b)]
    check("R0: found fenced bin/ commands to test", len(cmds) >= 8,
          "only %d found" % len(cmds))

    for cmd in cmds:
        argv = shlex.split(cmd)
        tool, rest = argv[0], argv[1:]
        if tool in PARSER_MODULES:
            # 'reaper' subcommand form parses through the same parser.
            err = try_parse(PARSER_MODULES[tool], rest)
            check("R1: parses against the real CLI: %s" % cmd[:76],
                  err is None, err or "")
        elif tool in WRAPPERS:
            wrapper = SUITE_ROOT / tool
            delegate = SUITE_ROOT / WRAPPERS[tool]
            import os
            check("R2: wrapper exists + executable + delegate present: %s" % tool,
                  wrapper.is_file() and os.access(str(wrapper), os.X_OK)
                  and delegate.is_file(),
                  "missing wrapper or delegate")
        else:
            check("R1: recipe names a tool this selftest knows: %s" % tool,
                  False, "add it to PARSER_MODULES or WRAPPERS")

    malformed = try_parse("measure_cloud", ["--max-cost", "not-a-decimal"])
    check("R6: malformed decimal is an argparse refusal, not a traceback",
          malformed == "argparse exit 2", malformed or "accepted")

    # Prose rungs for the recipe-2 fix. These FAIL on the pre-fix README.
    m = re.search(r"### Recipe 2 — local(.*?)### Recipe 3", section, re.S)
    r2 = m.group(1) if m else ""
    check("R3: Recipe 2 states the plan-only default and shows --execute",
          "plan-only by default" in r2 and "--execute" in r2)
    check("R4: Recipe 2 says measure-local downloads nothing + names the "
          "three local inputs",
          "downloads nothing" in r2 and "--artifact-path" in r2
          and "--teacher-tree" in r2 and "--pipeline-root" in r2)
    check("R5: Recipe 2 names the local lanes' surface limit",
          ("`packed`" in r2 and "`native-bf16`" in r2)
          or "support matrix" in r2)

    print()
    if failures:
        print("selftest_readme_recipes: %d FAILED" % len(failures))
        return 1
    print("selftest_readme_recipes: all rungs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
