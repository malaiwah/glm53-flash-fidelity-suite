<!--
SPLICE TARGET: the HF dataset card for malaiwah/quant-fidelity-registry
(README.md at the dataset root, below the YAML front matter).
Insert as a top-level section near the end, after the collections description.
Keep it self-contained: on HF this is often the only page a contributor reads.
-->

## Submit a measurement

**Discussions are the primary channel.** No git, no fork, no CI.

1. Run the measurement. Either runner in
   [malaiwah/quant-fidelity-suite](https://github.com/malaiwah/quant-fidelity-suite)
   seals a submission receipt for you — `bin/measure-cloud` on a rented GPU (it
   destroys the instance for you, on every exit path, and prints the real dollar
   cost), or `bin/measure-local` on your own Mac or CUDA box. Both write the
   receipt to `<out>/receipts/measurement-receipt.json`.

2. Verify the receipt sealed correctly. Four lines, no dependencies:

   ```python
   import json, hashlib
   d = json.load(open("submission.json")); claimed = d["receipt_sha256"]; d["receipt_sha256"] = ""
   canon = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
   print(hashlib.sha256(canon.encode()).hexdigest() == claimed)   # must print True
   ```

3. Open a **[new discussion](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/discussions)**
   titled `submission: <repo> on <panel>`, and paste:

   ````markdown
   ### Submission

   - **Artifact:** 0xSero/GLM-5.3-Flash-EXL3-Q4 @ 99cccdf0e8741715662c383828a9ea601990c125
   - **Panel:** panel--glm53.brandonmusic.final25
   - **Reference:** reference--brandonmusic.glm53-bf16-fp32-logits.final25
   - **Metric:** mean_of_run_means_tokenwise_kld = 0.027262784814670614 nats
   - **Lane:** sealed-ep8
   - **I am:** the measurer / also the quant's author? → measurer only
   - **Anything odd about this run:** none

   <details><summary>submission.json</summary>

   ```json
   { ...paste the whole file... }
   ```

   </details>
   ````

That is the whole submission. We validate it, generate the registry rows from
it, and reply in your thread with the row id, its comparability key, and which
existing rows yours can be compared against — or with exactly which check failed
and what to change.

**You are credited by HF handle.** The measurer of a number and the producer of
a quant are separate fields and neither is transferable: if you measured someone
else's quant, you get the number and they keep the quant. Your discussion URL is
attached to the row as its source, so anyone can find your submission and argue
with it.

**Prefer a pull request?** The GitHub mirror at
[malaiwah/quant-fidelity-registry](https://github.com/malaiwah/quant-fidelity-registry)
takes the same receipt as one file under `receipts/<your-handle>/`, with CI that
checks the seal, the schema and every registry invariant before a human looks.
Same result, more setup.

Full rules — mandatory fields, what gets bounced, how to register a new panel:
**[CONTRIBUTING.md](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/blob/main/CONTRIBUTING.md)**.
A real sealed example:
**[docs/examples/dione-q4.submission.json](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/blob/main/docs/examples/dione-q4.submission.json)**.
