# Measuring models whose modeling code ships in the repo

Approved by the maintainer, 2026-09-01.

## The situation this answers

Model vendors do not wait for `transformers`. At launch, an architecture is
served three ways that involve no transformers release at all: vLLM/SGLang
implement it natively in their own codebases; llama.cpp and MLX reimplement it
independently; and the repo itself ships `modeling_*.py` loaded via
`trust_remote_code=True`. Kimi-K3 is the live example: 1.5 TB, `kimi_k3`
absent from transformers 5.16.1, and a complete `modeling_kimi_k3.py` +
`auto_map` in the repo — loadable today by anyone willing to execute it.

Refusing all remote code forever would leave whole families unmeasurable for
weeks after launch, which is exactly when a fidelity root matters most (see
docs/RACE-MODE.md). Executing it casually would put arbitrary unaudited code in
the same process as our credentials and our claims.

## The policy

Remote code is acceptable **in the checkpoint lane only**, under four
conditions, all mandatory:

1. **Revision-pinned.** The model revision is a 40-hex commit, so the executed
   `.py` bytes are immutable and re-fetchable by anyone.
2. **Digested like our own code.** Every repo-shipped `.py` that can execute
   enters `harness.code_digests` with its sha256, alongside the suite's own
   estimator closure. The registry's whole claim is "we hashed what ran"; the
   origin of the code changes nothing about that obligation.
3. **Token-absent capture.** The HF token is needed for *fetch*, not for
   *capture*. The capture stage runs with the token unset and the 0600 token
   file already shredded from the environment the remote code can reach.
   Remote code that exfiltrates has nothing to exfiltrate.
4. **Disclosed on every row.** A `remote_code` disclosure
   (`affects_comparability: true`) on every measurement it produces.

## Enforcement, not convention

**RC-001** (schema/invariants.json, enforced in `registry_validate.py`,
selftest `remote-code-unrecorded-harness`): a row carrying a `remote_code`
disclosure with an **unrecorded harness** is refused as an error. A remote-code
row asserts "we executed code X"; an unrecorded harness asserts "we did not
hash what we executed". Together they are the one sentence this registry
exists to make unwritable.

## What this does not change

- The **serving lane** is untouched — vLLM's native implementations are that
  lane's identity, not a shortcut.
- `hy_v4`-style repos that ship **no** modeling code remain blocked on a
  transformers release; there is nothing to pin or digest.
- The generation sanity probe runs regardless: remote code that loads but
  produces a degenerate distribution fails the capture, not the reader.
