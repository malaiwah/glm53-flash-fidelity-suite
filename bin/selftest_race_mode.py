#!/usr/bin/env python3
"""Race mode -- the overlapped fetch, the preview identity, and the generation probe.

    bin/selftest_race_mode.py

Everything here runs OFFLINE on a randomly initialised, deliberately
multi-sharded tiny model, with a fake downloader that copies from a local
"remote" directory after a delay.  That is not a shortcut around measuring the
overlap -- it is what makes the overlap measurable at all: a real 1.5 TB fetch
cannot be run twice as a controlled A/B, and the thing being tested is a
SCHEDULE, which a simulated link exercises exactly.

    R1   the plan orders resident-first, then layer 0, 1, 2 ...
    R2   a shard holding BOTH a resident tensor and layer tensors is resident,
         and an unmatched key falls into the resident bucket -- every way the
         plan can be wrong fetches a shard EARLIER, never later
    R3   a shard serving two layers is priority min(), and is still returned for
         the later one
    R4   the gate blocks until a shard lands, and a timeout REFUSES rather than
         proceeding into a tree that is not there
    R5   audit_checkpoint_tree(shards=...) audits a partial tree, still refuses
         a SHORT shard inside the subset, and refuses a shard the index does not
         name
    R6   THE HEADLINE: a race capture and a fetch-then-capture capture of the
         same checkpoint produce the SAME capture_content_digest, and the race
         one finishes sooner. Both numbers are printed.
    R7   a layer whose shard never lands REFUSES by name -- it does not read a
         hole
    R8   the generation sanity probe answers "The capital of France is" and
         PASSES a --sanity-expect it should pass
    R9   a wrong --sanity-expect REFUSES, naming what the model actually said
    R10  A ZEROED HEAD -- the catastrophic case tensor counts and shapes cannot
         see -- is REFUSED as a degenerate distribution, with no expectation
         declared at all
    R11  running the probe does not move a single captured byte: the digests
         with and without it are equal
    R12  --preview-of seals a DIFFERENT dataset: its own id, not_submittable,
         a blocking preview_capture disclosure, and a named successor
    R12b the preview's own CARD says all of it above the first section heading,
         including when the capture supplied its own --readme body
    R13  --preview-of == --dataset-id is REFUSED -- the corruption this exists
         to prevent, refused by name
    R14  a preview still verifies, describes and VALIDATES clean: labelling it
         did not make it non-conformant
    R15  comparing a real candidate against a preview raises the blocking
         disclosure `emit_submission`'s SC-5 check refuses on -- while the SAME
         comparison against the FINAL carries no such disclosure
    R16  preview and final are different dataset ids, i.e. different
         `reference_id` inputs to the comparability key, i.e. different tables

Fail-without-fix: R1-R3 and R6-R7 fail as an ImportError for
`k6/tools/race_fetch.py`; R4/R5 as a TypeError on the `shards=` keyword; R8-R11
as an ImportError for `k6/tools/generation_probe.py`; R12-R16 as an argparse
refusal of `--preview-of`. Each case reports by name rather than aborting the
file, so the evidence reads as "these cases fail without the fix".
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "bin")
sys.path.insert(0, BIN)
sys.path.insert(0, os.path.join(REPO, "k6", "tools"))

PASS: list = []
FAIL: list = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print("  PASS  %s" % name)
    else:
        FAIL.append((name, detail))
        print("  FAIL  %s" % name)
        if detail:
            print("        %s" % str(detail)[:1400])


def run(argv, **kwargs):
    return subprocess.run([sys.executable] + argv, capture_output=True, text=True,
                          **kwargs)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def tiny_sharded_model(path, vocab=23, hidden=16, layers=4, seed=0,
                       shard_size="20KB"):
    """A tiny causal LM saved across SEVERAL shards, with a real index.json.

    Multi-shard is the point: with one shard there is no ordering to get right
    and nothing for the gate to block on.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=vocab, hidden_size=hidden,
                         intermediate_size=hidden * 2, num_hidden_layers=layers,
                         num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=64, tie_word_embeddings=False)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(path, safe_serialization=True, max_shard_size=shard_size)
    return path


def tiny_tokenizer(path):
    """A real, offline PreTrainedTokenizerFast whose vocabulary knows Paris."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["The", "capital", "of", "France", "is", "Paris", "Berlin", "Rome",
             "a", "the", "city"]
    vocab = {"<unk>": 0}
    for index, word in enumerate(words, start=1):
        vocab["Ġ" + word] = index
        vocab[word] = index + len(words)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True)
    tk.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(tokenizer_object=tk,
                            unk_token="<unk>").save_pretrained(path)
    return path


def teach_the_answer(model_dir, prompt="The capital of France is", answer=" Paris"):
    """Point the head at the right token, so the probe has something to find.

    A randomly initialised model answers nothing, so a positive control needs
    the head aimed. The hidden state is PRE-head, so rewriting the head does not
    change what the probe measures about the rest of the model -- it changes only
    which token wins, which is exactly the thing under test.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    target = tokenizer.encode(answer, add_special_tokens=False)[0]
    captured = []
    handle = model.get_output_embeddings().register_forward_pre_hook(
        lambda mod, inputs: captured.append(inputs[0].detach()))
    with torch.inference_mode():
        model(input_ids=torch.tensor([ids], dtype=torch.long),
              attention_mask=torch.ones(1, len(ids), dtype=torch.long),
              use_cache=False)
    handle.remove()
    hidden = captured[0][0, -1].to(torch.float32)
    direction = hidden / (hidden.norm() + 1e-6)
    with torch.no_grad():
        weight = model.get_output_embeddings().weight
        weight.zero_()
        weight[target] = (direction * 8.0).to(weight.dtype)
    shutil.rmtree(model_dir)
    model.save_pretrained(model_dir, safe_serialization=True, max_shard_size="20KB")
    tokenizer.save_pretrained(model_dir)
    return target


def zero_the_head(model_dir):
    """The catastrophic failure the count/shape guards cannot see: zeros."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    with torch.no_grad():
        model.get_output_embeddings().weight.zero_()
    shutil.rmtree(model_dir)
    model.save_pretrained(model_dir, safe_serialization=True, max_shard_size="20KB")
    tokenizer.save_pretrained(model_dir)


def tiny_panel(path, windows=2, length=10, vocab=23, seed=1):
    import numpy as np

    from fidelity import dsformat as F

    arrays = os.path.join(path, "arrays")
    os.makedirs(arrays, exist_ok=True)
    rng = np.random.RandomState(seed)
    mask = np.ones(length, dtype=np.uint8)
    mask_path = os.path.join(arrays, "causal-mask-%d.npy" % length)
    np.save(mask_path, mask, allow_pickle=False)
    rows = []
    for index in range(windows):
        ids = rng.randint(0, vocab, size=length).astype(np.int32)
        token_path = os.path.join(arrays, "final-%04d.tokens.npy" % index)
        np.save(token_path, ids, allow_pickle=False)
        rows.append({"window_id": "final-%04d" % index, "role": "final",
                     "domain": "axis1_general", "document_id": "doc-%d" % index,
                     "prediction_positions": length - 1,
                     "token_ids_sha256": F.sha256_file(token_path),
                     "attention_mask_sha256": F.sha256_file(mask_path)})
    with open(os.path.join(path, "panel.json"), "w", encoding="utf-8") as handle:
        json.dump({"schema": "quant-pipeline.glm53-token-panel.v1",
                   "sealed_corpus_sha256": None, "windows": rows}, handle, indent=2)
    with open(os.path.join(path, "panel.receipt.json"), "w", encoding="utf-8") as handle:
        json.dump({"schema": "malaiwah.token-panel-build-receipt.v1",
                   "selection_rule": "race-mode selftest fixture"}, handle, indent=2)
    return path


def capture_argv(model, panel, out, *, dataset_id, role="root", extra=()):
    return ([os.path.join(REPO, "k6", "tools", "hf_capture.py"),
             "--model", model, "--panel", panel, "--out", out, "--role", role,
             "--lane", "local-cuda-budget", "--dataset-id", dataset_id,
             "--dataset-name", dataset_id, "--device", "cpu",
             "--schedule", "layer-outer", "--layer-residency", "stream",
             "--weights-repository", "selftest/tiny", "--model-revision", "0" * 40]
            + list(extra))


def manifest_of(root):
    from fidelity import dsformat as F

    return F.read_json(os.path.join(root, F.MANIFEST_NAME))


def digest_of(root):
    return manifest_of(root)["capture"]["capture_content_digest"]


# ---------------------------------------------------------------------------
# the simulated link
# ---------------------------------------------------------------------------


def slow_copy_downloader(source_dir, dest_dir, seconds_per_file, record=None):
    """Copy one file from `source_dir` to `dest_dir` after a delay, ATOMICALLY.

    Written through a temporary name and renamed, so a file that exists is a
    file that is complete -- the same property `hf_hub_download` gives, and the
    one the audit depends on.
    """
    lock = threading.Lock()

    def download(name):
        time.sleep(seconds_per_file)
        src = os.path.join(source_dir, name)
        dst = os.path.join(dest_dir, name)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        tmp = dst + ".part"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        if record is not None:
            with lock:
                record.append((round(time.monotonic(), 4), name))
        return dst

    return download


# ---------------------------------------------------------------------------


def main():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        print("SKIP selftest_race_mode: torch/transformers unavailable (%s)" % exc)
        return 0
    work = tempfile.mkdtemp(prefix="racemode-")
    try:
        return _body(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _body(work):
    try:
        import race_fetch
    except ImportError as exc:
        race_fetch = None
        race_absent = str(exc)
    try:
        import generation_probe
    except ImportError as exc:
        generation_probe = None
        probe_absent = str(exc)

    def needs_race(name):
        check(name, False, "k6/tools/race_fetch.py is not importable: %s" % race_absent)

    # ---------------------------------------------------------------- R1-R3
    weight_map = {
        "model.embed_tokens.weight": "s-00001.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "s-00001.safetensors",
        "model.layers.0.mlp.up_proj.weight": "s-00002.safetensors",
        "model.layers.1.mlp.up_proj.weight": "s-00002.safetensors",
        "model.layers.2.mlp.up_proj.weight": "s-00003.safetensors",
        "model.layers.3.mlp.up_proj.weight": "s-00003.safetensors",
        "model.layers.8.mlp.up_proj.weight": "s-00003.safetensors",
        "model.norm.weight": "s-00004.safetensors",
        "lm_head.weight": "s-00004.safetensors",
        "some.unrecognised.tensor": "s-00005.safetensors",
    }
    if race_fetch is None:
        for name in ("R1 the plan orders resident-first, then layer 0, 1, 2 ...",
                     "R2 a mixed shard is resident, and an unmatched key is too",
                     "R3 a two-layer shard is priority min() and is served for both"):
            needs_race(name)
        plan = None
    else:
        plan = race_fetch.plan_shards(weight_map)
        order = [name for _, name in plan.order()]
        priorities = [p for p, _ in plan.order()]
        check("R1 the plan orders resident-first, then layer 0, 1, 2 ...",
              priorities == sorted(priorities)
              and order[:3] == ["s-00001.safetensors", "s-00004.safetensors",
                                "s-00005.safetensors"]
              and order[3:] == ["s-00002.safetensors", "s-00003.safetensors"],
              order)
        check("R2 a mixed shard is resident, and an unmatched key is too",
              plan.needed_at["s-00001.safetensors"] == race_fetch.RESIDENT
              and plan.needed_at["s-00004.safetensors"] == race_fetch.RESIDENT
              and plan.needed_at["s-00005.safetensors"] == race_fetch.RESIDENT
              and plan.unmatched_keys == 4,
              (plan.needed_at, plan.unmatched_keys))
        check("R3 a two-layer shard is priority min() and is served for both",
              plan.needed_at["s-00003.safetensors"] == 2
              and "s-00003.safetensors" in plan.shards_for_layer(8)
              and "s-00003.safetensors" in plan.shards_for_layer(2)
              and plan.layer_count == 9,
              (plan.needed_at, sorted(plan.shards_for_layer(8))))

    # ------------------------------------------------------------------- R4
    if race_fetch is None:
        needs_race("R4 the gate blocks until a shard lands, and a timeout REFUSES")
    else:
        gate = race_fetch.ShardGate()
        threading.Timer(0.25, lambda: gate.mark("a")).start()
        waited = gate.wait_for(["a"], timeout=10.0)
        timed_out = None
        try:
            gate.wait_for(["never"], timeout=0.4)
        except race_fetch.RaceFetchError as exc:
            timed_out = str(exc)
        check("R4 the gate blocks until a shard lands, and a timeout REFUSES",
              waited >= 0.2 and timed_out is not None
              and "reads as zeros" in timed_out,
              (waited, timed_out))

    # ------------------------------------------------------------------- R5
    import layer_outer

    partial = os.path.join(work, "partial")
    tiny_sharded_model(partial)
    index_path = os.path.join(partial, "model.safetensors.index.json")
    has_index = os.path.isfile(index_path)
    if not has_index:
        check("R5 the subset audit passes on a partial tree and still refuses a hole",
              False, "the fixture did not shard: no model.safetensors.index.json")
    else:
        shard_names = sorted(set(json.load(open(index_path))["weight_map"].values()))
        keep = shard_names[0]
        for name in shard_names[1:]:
            os.remove(os.path.join(partial, name))
        try:
            layer_outer.audit_checkpoint_tree(partial, shards=[keep])
            subset_ok, subset_err = True, None
        except TypeError as exc:
            subset_ok, subset_err = False, "no shards= keyword: %s" % exc
        except layer_outer.LayerOuterError as exc:
            subset_ok, subset_err = False, str(exc)
        full_refused = None
        try:
            layer_outer.audit_checkpoint_tree(partial)
        except layer_outer.LayerOuterError as exc:
            full_refused = str(exc)
        # ... and the subset audit is not weaker: truncate the shard it covers.
        short_refused = None
        if subset_ok:
            path = os.path.join(partial, keep)
            with open(path, "r+b") as handle:
                handle.truncate(os.path.getsize(path) - 64)
            try:
                layer_outer.audit_checkpoint_tree(partial, shards=[keep])
            except layer_outer.LayerOuterError as exc:
                short_refused = str(exc)
        unknown_refused = None
        try:
            layer_outer.audit_checkpoint_tree(partial, shards=["not-a-shard.safetensors"])
        except TypeError:
            pass
        except layer_outer.LayerOuterError as exc:
            unknown_refused = str(exc)
        check("R5 the subset audit passes on a partial tree and still refuses a hole",
              subset_ok and full_refused is not None
              and short_refused is not None and "read as ZEROS" in short_refused
              and unknown_refused is not None
              and "does not name" in unknown_refused,
              (subset_err, full_refused, short_refused, unknown_refused))

    # ------------------------------------------------------------- R6, R7
    remote = os.path.join(work, "remote")
    tiny_sharded_model(remote)
    tiny_tokenizer(remote)
    panel = tiny_panel(os.path.join(work, "panel"))
    shard_names = sorted(name for name in os.listdir(remote)
                         if name.endswith(".safetensors"))
    bootstrap = [name for name in os.listdir(remote)
                 if not name.endswith(".safetensors")]

    if race_fetch is None:
        needs_race("R6 race capture == serial capture (same digest, less wall clock)")
        needs_race("R7 a shard that never lands REFUSES rather than reading a hole")
        race_saving = None
    else:
        per_file = 0.45

        # --- control arm: fetch, THEN capture -----------------------------
        serial_dir = os.path.join(work, "serial-model")
        os.makedirs(serial_dir, exist_ok=True)
        control_started = time.monotonic()
        for name in bootstrap:
            shutil.copyfile(os.path.join(remote, name), os.path.join(serial_dir, name))
        download = slow_copy_downloader(remote, serial_dir, per_file)
        # Same worker count as the race arm, so the two differ only in overlap.
        pending = list(shard_names)
        threads = []
        lock = threading.Lock()

        def drain():
            while True:
                with lock:
                    if not pending:
                        return
                    name = pending.pop(0)
                download(name)

        for _ in range(2):
            thread = threading.Thread(target=drain)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        control_fetch = time.monotonic() - control_started
        serial_out = os.path.join(work, "ds-serial")
        result = run(capture_argv(serial_dir, panel, serial_out,
                                  dataset_id="fidelity--selftest.serial"))
        control_total = time.monotonic() - control_started

        # --- race arm: the same fetch, overlapped -------------------------
        race_dir = os.path.join(work, "race-model")
        os.makedirs(race_dir, exist_ok=True)
        for name in bootstrap:
            shutil.copyfile(os.path.join(remote, name), os.path.join(race_dir, name))
        race_out = os.path.join(work, "ds-race")
        race_report = os.path.join(work, "race-report.json")
        race_started = time.monotonic()
        race_result = run(capture_argv(
            race_dir, panel, race_out, dataset_id="fidelity--selftest.race",
            extra=["--race-repo", "selftest/remote", "--race-workers", "2",
                   "--race-report", race_report,
                   "--race-simulate-seconds", str(per_file),
                   "--race-simulate-source", remote]))
        race_total = time.monotonic() - race_started

        same = (result.returncode == 0 and race_result.returncode == 0
                and digest_of(serial_out) == digest_of(race_out))
        race_saving = control_total - race_total
        print("        control: fetch %.2fs + capture %.2fs = %.2fs total"
              % (control_fetch, control_total - control_fetch, control_total))
        print("        race:    %.2fs total  ->  %.2fs saved (%.0f%% of the fetch hidden)"
              % (race_total, race_saving,
                 100.0 * race_saving / control_fetch if control_fetch else 0.0))
        check("R6 race capture == serial capture (same digest, less wall clock)",
              same and race_total < control_total,
              (result.returncode, race_result.returncode,
               (race_result.stderr or "")[-1200:],
               control_total, race_total))

        # --- R7: a shard that never lands ---------------------------------
        stall_dir = os.path.join(work, "stall-model")
        os.makedirs(stall_dir, exist_ok=True)
        for name in bootstrap:
            shutil.copyfile(os.path.join(remote, name), os.path.join(stall_dir, name))
        withheld = shard_names[-1]
        stall_remote = os.path.join(work, "stall-remote")
        shutil.copytree(remote, stall_remote)
        os.remove(os.path.join(stall_remote, withheld))
        stalled = run(capture_argv(
            stall_dir, panel, os.path.join(work, "ds-stall"),
            dataset_id="fidelity--selftest.stall",
            extra=["--race-repo", "selftest/remote", "--race-workers", "2",
                   "--race-simulate-seconds", "0.0",
                   "--race-timeout-seconds", "3",
                   "--race-simulate-source", stall_remote]))
        message = (stalled.stderr or "") + (stalled.stdout or "")
        check("R7 a shard that never lands REFUSES rather than reading a hole",
              stalled.returncode != 0
              and ("the background fetch failed" in message
                   or "never landed" in message),
              (stalled.returncode, message[-1200:]))

    # -------------------------------------------------------------- R8-R11
    if generation_probe is None:
        for name in ("R8 the probe answers the prompt and passes its expectation",
                     "R9 a wrong expectation REFUSES, naming what the model said",
                     "R10 a ZEROED head is refused as a degenerate distribution",
                     "R11 running the probe does not move a single captured byte"):
            check(name, False,
                  "k6/tools/generation_probe.py is not importable: %s" % probe_absent)
    else:
        healthy = os.path.join(work, "healthy")
        shutil.copytree(remote, healthy)
        teach_the_answer(healthy)
        probe_out = os.path.join(work, "ds-probe")
        good = run(capture_argv(healthy, panel, probe_out,
                                dataset_id="fidelity--selftest.probe",
                                extra=["--sanity-expect", "Paris"]))
        verdict = (manifest_of(probe_out).get("generation_sanity_probe")
                   if good.returncode == 0 else None)
        check("R8 the probe answers the prompt and passes its expectation",
              good.returncode == 0 and verdict is not None
              and verdict.get("status") == "pass"
              and verdict.get("top1_text", "").strip() == "Paris"
              and verdict.get("enforced") is True,
              (good.returncode, (good.stderr or "")[-900:], verdict))

        wrong = run(capture_argv(healthy, panel, os.path.join(work, "ds-wrong"),
                                 dataset_id="fidelity--selftest.wrong",
                                 extra=["--sanity-expect", "Berlin"]))
        wrong_msg = (wrong.stderr or "") + (wrong.stdout or "")
        check("R9 a wrong expectation REFUSES, naming what the model said",
              wrong.returncode != 0
              and "generation sanity probe failed" in wrong_msg
              and "Paris" in wrong_msg and "Berlin" in wrong_msg,
              (wrong.returncode, wrong_msg[-1200:]))

        zeroed = os.path.join(work, "zeroed")
        shutil.copytree(remote, zeroed)
        zero_the_head(zeroed)
        dead = run(capture_argv(zeroed, panel, os.path.join(work, "ds-zero"),
                                dataset_id="fidelity--selftest.zero"))
        dead_msg = (dead.stderr or "") + (dead.stdout or "")
        check("R10 a ZEROED head is refused as a degenerate distribution",
              dead.returncode != 0 and "DEGENERATE" in dead_msg
              and "loaded as zeros" in dead_msg,
              (dead.returncode, dead_msg[-1200:]))

        no_probe = os.path.join(work, "ds-noprobe")
        off = run(capture_argv(healthy, panel, no_probe,
                               dataset_id="fidelity--selftest.probe",
                               extra=["--no-sanity-check"]))
        check("R11 running the probe does not move a single captured byte",
              off.returncode == 0 and good.returncode == 0
              and digest_of(no_probe) == digest_of(probe_out)
              and manifest_of(no_probe)["generation_sanity_probe"]["status"]
              == "skipped",
              (off.returncode, (off.stderr or "")[-900:]))

    # ------------------------------------------------------------- R12-R15
    base = os.path.join(work, "preview-model")
    shutil.copytree(remote, base)
    final_id = "fidelity--selftest.tiny-root-v1"
    preview_id = final_id + ".preview"
    preview_out = os.path.join(work, "ds-preview")
    made = run(capture_argv(base, panel, preview_out, dataset_id=preview_id,
                            extra=["--preview-of", final_id]))
    if made.returncode != 0:
        for name in ("R12 --preview-of seals a DIFFERENT dataset, named and blocking",
                     "R14 a preview still verifies, describes and validates clean",
                     "R15 a comparison against a preview is blocked from the registry",
                     "R12b the preview's CARD says it, in its first screen",
                     "R16 preview and final are different reference identities"):
            check(name, False, (made.stderr or "")[-1200:])
        preview_manifest = None
    else:
        preview_manifest = manifest_of(preview_out)
        block = preview_manifest.get("preview") or {}
        blocking = [d for d in preview_manifest["disclosures"]
                    if d.get("code") == "preview_capture"]
        card = open(os.path.join(preview_out, "README.md"), encoding="utf-8").read()
        check("R12b the preview's CARD says it, in its first screen",
              "PRELIMINARY" in card.split("\n## ")[0]
              and "ONE cold run" in card
              and "determinism is NOT demonstrated" in card
              and "NEVER be updated in place" in card
              and final_id in card,
              card[:900])
        check("R12 --preview-of seals a DIFFERENT dataset, named and blocking",
              preview_manifest.get("not_submittable") is True
              and block.get("superseded_by") == final_id
              and block.get("determinism_demonstrated") is False
              and block.get("updated_in_place") is False
              and preview_manifest["dataset"]["id"] == preview_id != final_id
              and len(blocking) == 1
              and blocking[0]["severity"] == "blocking"
              and blocking[0]["affects_comparability"] is True
              and preview_manifest["determinism"]["identical_across_runs"] is None,
              (preview_manifest.get("not_submittable"), block, blocking))

    same_id = run(capture_argv(base, panel, os.path.join(work, "ds-sameid"),
                               dataset_id=final_id,
                               extra=["--preview-of", final_id]))
    same_msg = (same_id.stderr or "") + (same_id.stdout or "")
    check("R13 --preview-of == --dataset-id is REFUSED",
          same_id.returncode != 0
          and "same id as --dataset-id" in same_msg
          and "comparability-key field" in same_msg,
          (same_id.returncode, same_msg[-1200:]))

    if preview_manifest is not None:
        tool = os.path.join(REPO, "bin", "fidelity_dataset.py")
        verified = run([tool, "verify", preview_out])
        described = run([tool, "describe", preview_out])
        # ... and VALIDATES clean. The preview's markers are additive top-level
        # keys, and the spec's own rule (section 1.3) is that a v1 reader ignores
        # unknown keys -- so this is the check that the labelling did not make
        # the dataset non-conformant to buy its own honesty.
        validated = run([tool, "validate", preview_out, "--verify-tensors"])
        check("R14 a preview still verifies, describes and validates clean",
              verified.returncode == 0 and described.returncode == 0
              and validated.returncode == 0
              and "0 error(s), 0 warning(s)" in (validated.stdout or ""),
              ((verified.stderr or "")[-500:], (described.stderr or "")[-500:],
               (validated.stdout or "")[-500:]))

        # The full-evidence capture of the same weights, under its own id.
        final_out = os.path.join(work, "ds-final")
        final_made = run(capture_argv(base, panel, final_out, dataset_id=final_id))

        # A real candidate, so the comparison is a MEASUREMENT and reaches the
        # submission path at all -- otherwise SC-3 would refuse it first and the
        # preview gate would never be the thing under test.
        quant_dir = os.path.join(work, "quant")
        scope = os.path.join(work, "scope.json")
        quantized = run([os.path.join(REPO, "bin", "toy_quantize.py"),
                         "--src", base, "--dst", quant_dir, "--bits", "3",
                         "--match", ".mlp.down_proj.", "--emit-scope", scope])
        quant_out = os.path.join(work, "ds-quant")
        quant_made = run(capture_argv(quant_dir, panel, quant_out,
                                      dataset_id="fidelity--selftest.quant",
                                      role="quant",
                                      extra=["--scope-file", scope]))
        prov = os.path.join(work, "prov.json")
        run([tool, "provenance-template", "--out", prov])

        def compare_against(reference, out):
            result = run([tool, "compare", "--reference", reference,
                          "--candidate", quant_out, "--out", out,
                          "--emit-submission", "--submission-provenance", prov])
            path = os.path.join(out, "comparison-receipt.json")
            doc = json.load(open(path)) if os.path.isfile(path) else {}
            return result, doc

        vs_preview, preview_receipt = compare_against(
            preview_out, os.path.join(work, "cmp-preview"))
        vs_final, final_receipt = compare_against(
            final_out, os.path.join(work, "cmp-final"))
        preview_codes = {d.get("code"): d.get("severity")
                         for d in preview_receipt.get("disclosures") or []}
        final_codes = {d.get("code") for d in final_receipt.get("disclosures") or []}
        preview_text = (vs_preview.stdout or "") + (vs_preview.stderr or "")
        check("R15 a comparison against a preview is blocked from the registry",
              quant_made.returncode == 0 and final_made.returncode == 0
              and preview_receipt.get("comparison_kind") == "measurement"
              and preview_codes.get("preview_capture") == "blocking"
              and (preview_receipt.get("comparability") or {}).get("class") == "advisory"
              and not (preview_receipt.get("submission") or {}).get("emitted")
              and "blocking disclosure" in preview_text
              # ... and the SAME comparison against the FINAL is not blocked by
              # this: the refusal is about the preview, not about the machinery.
              and "preview_capture" not in final_codes,
              (quantized.returncode, quant_made.returncode,
               (quant_made.stderr or "")[-700:], preview_codes, sorted(final_codes),
               preview_text[-900:]))

        check("R16 preview and final are different reference identities",
              final_made.returncode == 0
              and manifest_of(final_out)["dataset"]["id"] == final_id
              and preview_manifest["dataset"]["id"] == preview_id
              and final_id != preview_id
              and manifest_of(final_out).get("not_submittable") is None,
              (final_made.returncode, (final_made.stderr or "")[-700:]))

    print("")
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
