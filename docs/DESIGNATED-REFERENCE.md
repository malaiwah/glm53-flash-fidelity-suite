# When a family publishes no unquantized weights

Every measurement in this registry is a distance **from** something. Until now
that something was always the model's own unquantized release, and a family
that publishes none was simply unmeasurable: Stage A closed
`deepseek-ai/DeepSeek-V4-Flash-0731` as NO-GO for exactly this reason, with a
working engine and no anchor. **Every** `deepseek_v4` repo on the Hub ships a
`quantization_config` — the published root *is* FP8-E4M3 attention plus
FP4-E2M1 experts — and `mlx-community/DeepSeek-V4-Flash-bf16` is a 4-bit MLX
repo wearing a misleading name.

That is 100 quantized children and 4.58M downloads with no yardstick, because
of a missing artifact nobody is going to publish.

## The rule

> When no unquantized release of a model exists, the published artifact with
> the **most bits — actual, never extrapolated** — is DESIGNATED as that
> family's reference.

Three things it does **not** change, and one it does:

* It does not change any **capture**. The bytes read off a checkpoint are the
  same bytes either way.
* It does not change any **fidelity dataset**. A dataset records what a model
  computed on a panel; the designation is about what other numbers are compared
  against, which is a later and separate step.
* It does not change any **artifact record**. The proxy is still described as
  exactly the quantized thing it is.
* It changes only the **measures** — what the divergences are distances from.

And it is **superseded, never invalidated**. If true unquantized weights are
released later, rows measured against them carry a new `reference_id` and
therefore land in a new comparability group. The proxy-referenced rows remain
correct statements about what they actually measured.

## Why this is safe, and how that is enforced rather than promised

`COMPARABILITY_KEY_FIELDS` binds `reference_id`. So a row against a designated
proxy **cannot** share a comparability group with a row against unquantized
weights — not by convention, but because the key is a hash over inputs that
include the reference identity. Demonstrated rather than asserted:

```
BF16-referenced key : cmp--209383798edd8dc2
proxy-referenced key: cmp--092d2cc380b22bab
```

Identical in every other field. The registry could not rank them together if
it wanted to.

What the key cannot do is stop a **reader** from quoting a proxy-referenced
number as though it were divergence from the model. So **REFC-006** requires,
as errors:

1. `reference_kind = quantized_proxy` → the referenced artifact must have
   `kind` in `{quant, requantized}`. A proxy pointing at a base artifact is a
   `native_*` reference wearing the wrong label, and would escape (2) and (3).
2. Every measurement against it carries a `different_reference_kind`
   disclosure with `affects_comparability: true`.
3. Every such measurement is `comparability.class = advisory`. **A designated
   proxy is not a measured floor.**

Three selftest cases prove each clause fires:
`proxy-reference-undisclosed`, `proxy-reference-marked-strict`,
`proxy-reference-on-base-artifact`.

## What the number means, stated plainly

A row against a designated proxy answers *"how far is this quant from the best
published version of this model?"* — **not** *"how far is this quant from the
model?"*.

It is systematically **smaller** than the true divergence, because the proxy
already carries its own quantization error and the child inherits it rather
than being charged for it. The reference's own self-compare is exactly 0.0 by
construction, and that 0.0 is a **designation, not a floor**: it says "this is
the origin we chose", not "this model was measured to lose nothing".

The existing `reference_kind` vocabulary already said this, in the schema's own
words, before there was any tooling for it:

> `dequantized_from_quant` means the reference distribution is itself a
> quantized model's, so student numbers against it are systematically SMALLER
> than against a true BF16 teacher and must never be ranked against `native_*`
> rows.

`quantized_proxy` is the sibling case where the quantized artifact is used
**directly** as the teacher rather than dequantized first. It was in the enum
and had no invariant; REFC-006 is that invariant.

## Choosing the proxy

"Most bits, actual, never extrapolated" means: read the rates the release
itself publishes, as `--scope-json` already requires and as the plan-time gate
already verifies against the artifact's own `quantization_config`. A nominal
bpw in a repo name is not evidence. Where two candidates are close, prefer the
one whose scope is **fully read** over one with `unknown` classes — an
unmeasurable recipe makes a poor origin.

For `deepseek_v4` the choice is degenerate and therefore easy: the family's
own published root is the most-faithful artifact in it, and every other repo is
a quantization of that. The reference is the root, the children are its
quantizations, and the engine to read them is already proven — 3,176/3,176
tensors byte-exact, self-compare exactly 0.0.
