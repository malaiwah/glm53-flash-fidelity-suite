#!/usr/bin/env python3
"""Selftest: every fenced command in the recipe documents is executable as
written — it parses against the real CLI it names, the prose makes the
plan-only/cannot-fetch limits explicit, and the paid root recipes' numbers
are not below the bound the controller computes for their target.

The defect this guards against (contributor-experience review, 2026-08-31):
the local recipe was **not executable as written**. It omitted `--execute`
(measure-local is plan-only by default and exits 3 on a clean plan), and said
nothing about measure-local being unable to fetch the artifact, the teacher or
the pipeline — so a stranger pasted it, got a refusal, and could not tell
whether they had broken something. Docs had also drifted from argparse before
(the 2026-08 flag reconciliation found five documented flags that did not
exist), so this test PROBES the CLI rather than trusting the prose, the same
rule the engine pinning uses.

The second defect (cloud usability review, 2026-09-05, S1-2): four copies of
the root recipe shipped `--max-cost 40 --max-runtime 3h30m` after the bound in
`bin/engines.json` had been re-authored to 26925 s, so the copy/paste recipe
refused on its own numbers twice before showing a plan. R7 recomputes the
bound with the controller's own arithmetic (`measure_cloud._root_workload_bound`)
for every documented root recipe whose target has a timing row, and R8 holds
`--max-cost` at or above the all-in figure the same documents state.

Rungs:
  R1  every `bin/<tool>` command in a ```bash fence of README's recipe
      sections, docs/THIRD-PARTY-QUICKSTART.md, docs/CLOUD-RECIPES.md and
      registry/CONTRIBUTING.md parses against that tool's real argparse
      parser (placeholders substituted; unknown flags or missing required
      args fail the rung).
  R2  wrapper commands with no importable parser (registry-submit) exist and
      are executable, and their delegate script exists.
  R3  Recipe 2 states the plan-only default and shows `--execute`.
  R4  Recipe 2 states that measure-local downloads nothing and names all
      three local inputs (--artifact-path, --teacher-tree, --pipeline-root).
  R5  Recipe 2 names the local lanes' surface limit (packed / native-bf16)
      or links to the support matrix.
  R6  a malformed decimal is an argparse refusal, not a traceback.
  R7  every documented `measure-cloud --role root` recipe with a 40-hex
      revision whose target has an authored timing row carries
      `--max-runtime` >= the controller's workload bound (fresh root, two
      captures, container-disk layout).
  R8  the same recipes carry `--max-cost` >= the all-in maximum the cloud
      documents state for that target, and the documents agree on it.

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
RECIPE_DOCS = (
    SUITE_ROOT / "docs" / "THIRD-PARTY-QUICKSTART.md",
    SUITE_ROOT / "docs" / "CLOUD-RECIPES.md",
    SUITE_ROOT / "registry" / "CONTRIBUTING.md",
)
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
    "<hf-url>": "example-org/example-quant",
    "<out>": "/tmp/example-out",
    "<out-A>": "/tmp/example-out-a",
    "<receipt.json>": "/tmp/example-out/receipts/measurement-receipt.json",
    "<panel>": "example-org/example-panel",
    "<rev>": "0" * 40,
    "<40-hex>": "0" * 40,
    "<40hex>": "0" * 40,
    "<revision>": "0" * 40,
    "<your-hf-handle>": "example-handle",
    "<hub-handle>": "example-handle",
    "<token>": "EXAMPLE",
    "<owner>/<repo>": "example-org/example-root",
    "<owner>/<quant>": "example-org/example-quant",
    "<owner>/<repo>@<revision>": "example-org/example-root@" + "0" * 40,
    "<owner>": "example-org",
    "<repo>": "example-root",
    "<quant>": "example-quant",
    "<id>": "example-id",
    "<name>": "example-name",
    "<slug>": "example-slug",
    "<scope.json>": "/tmp/example-scope.json",
    "<dir>": "/tmp/example-dir",
    "<handle>": "example-handle",
    "<usd>": "10",
    "<duration>": "1h",
    "<build>": "example-build",
    "<lane>": "streaming",
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
    "$ROOT_ATTEMPT_CAP_USD": "40",
    "$ROOT_MAX_RUNTIME": "3h30m",
    "$ROOT_RETRIEVAL_DELETE_RESERVE_SECONDS": "14400",
    "$HOME": "/tmp/example-home",
}

# argv[0] -> module with build_parser(). Probed, never guessed.
PARSER_MODULES = {
    "bin/measure": "measure_one",
    "bin/measure-cloud": "measure_cloud",
    "bin/measure-local": "measure_local",
    "bin/registry-view": "registry_view",
    "bin/fidelity-card": "fidelity_card",
}

# Wrappers whose contract is checked structurally (no importable parser).
WRAPPERS = {
    "bin/registry-submit": "registry/tools/registry_validate.py",
    "bin/fidelity-doctor": "bin/fidelity-doctor",
    "bin/fixture": "bin/fixture_fetch.py",
    "bin/fidelity-bench": "bin/fidelity/bench.py",
    "bin/fidelity-post": "bin/fidelity_post.py",
    "bin/fidelity-dataset": "bin/fidelity_dataset.py",
}

# The all-in figure the cloud documents state for a root target, and the
# rate they quote it at -- read from the prose so the recipes and the prose
# cannot disagree silently.
ALL_IN_RE = re.compile(r"\$(\d+\.\d+)\s+at\s+(?:today's\s+)?\$(\d+\.\d+)/h")


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
    for key in sorted(PLACEHOLDERS, key=len, reverse=True):
        cmd = cmd.replace(key, PLACEHOLDERS[key])
    # "$(...)" command substitutions: a recipe reads a digest or a byte count
    # out of a receipt. The parser sees a value of the right shape.
    cmd = re.sub(r'"\$\(stat [^)]*\)"', "1", cmd)
    cmd = re.sub(r'"\$\((?:[^()]|\([^()]*\))*\)"', "0" * 64, cmd)
    return cmd


def flag_value(argv, flag):
    for i, token in enumerate(argv[:-1]):
        if token == flag:
            return argv[i + 1]
    return None


def root_recipe_bounds(cmds):
    """(cmd, bound_seconds, max_runtime_seconds, max_cost, target) for every
    documented root recipe whose target has an authored timing row.

    A recipe that writes `--revision <40-hex>` still names a real target; it
    is held to every authored row for that repo (the README and CLOUD-RECIPES
    copies were the ones that went stale)."""
    import measure_cloud
    from fidelity.common import parse_duration
    from fidelity.engines import _load_engine_config
    from decimal import Decimal
    authored = [row for row in _load_engine_config().get("root_timing_profiles") or []
                if isinstance(row, dict)]
    rows = []
    for cmd in cmds:
        argv = shlex.split(cmd)
        if argv[0] != "bin/measure-cloud" or flag_value(argv, "--role") != "root":
            continue
        if "--candidate-scope" in argv or "--resume-capture" in argv:
            continue
        repo, rev = flag_value(argv, "--model"), flag_value(argv, "--revision")
        gpu = flag_value(argv, "--gpu")
        placeholder = rev == PLACEHOLDERS["<rev>"]
        matches = [row for row in authored
                   if row.get("target_repo") == repo
                   and (placeholder or row.get("target_revision") == rev)
                   and (gpu is None or row.get("gpu") == gpu)
                   and row.get("form") == "hidden"
                   and row.get("schedule") == "two-fresh-process-qualification"]
        if not matches:
            continue
        runtime = parse_duration(flag_value(argv, "--max-runtime") or "0")
        cost = Decimal(flag_value(argv, "--max-cost") or "0")
        for timing in matches:
            try:
                bound, _ = measure_cloud._root_workload_bound(
                    timing,
                    storage_layout=flag_value(argv, "--storage-layout") or "container-disk",
                    captures=2)
            except Exception as exc:                          # noqa: BLE001
                rows.append((cmd, None, None, None, "%s (%s)" % (repo, exc)))
                continue
            rows.append((cmd, int(bound), runtime, cost,
                         "%s@%s" % (repo, timing["target_revision"][:12])))
    return rows


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
    # The quickstart, the cloud recipes and the contributor guide are recipe
    # documents too — same contract: executable as written, or explicitly
    # $0/refusing.
    doc_texts = {}
    for doc in RECIPE_DOCS:
        if doc.is_file():
            doc_texts[doc] = doc.read_text(encoding="utf-8")
            blocks += bash_blocks(doc_texts[doc])
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

    # Numeric tripwires for the paid root recipes (S1-2). These FAIL on the
    # pre-fix docs (--max-runtime 3h30m against a 26925 s bound).
    bounds = root_recipe_bounds(cmds)
    check("R7: found root recipes with an authored timing row", len(bounds) >= 4,
          "%d found" % len(bounds))
    stated = {}
    for doc, text in doc_texts.items():
        for all_in, rate in ALL_IN_RE.findall(text):
            stated.setdefault(all_in, set()).add(doc.name)
    for cmd, bound, runtime, cost, target in bounds:
        if bound is None:
            check("R7: timing lookup for %s" % target, False, "lookup raised")
            continue
        check("R7: --max-runtime %ds >= bound %ds for %s: %s"
              % (runtime, bound, target, cmd[:60]), runtime >= bound)
        if "GLM-5.3-BF16" in target:
            check("R8: cloud docs state one all-in figure for the GLM-5.3 root",
                  len(stated) == 1, "stated: %s" % sorted(stated))
            from decimal import Decimal
            for all_in in stated:
                check("R8: --max-cost %s >= stated all-in $%s: %s"
                      % (cost, all_in, cmd[:60]), cost >= Decimal(all_in))
    print()
    if failures:
        print("selftest_readme_recipes: %d FAILED" % len(failures))
        return 1
    print("selftest_readme_recipes: all rungs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
