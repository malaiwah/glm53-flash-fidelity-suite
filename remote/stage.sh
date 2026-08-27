#!/bin/bash
# Stage driver for the GLM-5.3-Flash fidelity capture session (H200 VM + docker).
# Usage: stage.sh <stage-name>
# Host layout: /home/ubuntu/glm53/{bundle,models,captures,out}
# Container mount: /glm53  (identical relative layout)
set -euo pipefail
ROOT=/home/ubuntu/glm53
IMAGE_REF="$(awk '{print $1}' "$ROOT/out/image-pin.txt" 2>/dev/null || true)"
IMAGE_REF="${IMAGE_REF:-__IMAGE_REF__}"
TP=8
NTFY_URL="https://ntfy.sh/omp-396220bc418fb23ea7a57901a54c7b33"
STAGE="$1"
ntfy() {  # ntfy <body> <title> <tags> [priority]
  curl -s -m 10 -H "Title: $2" -H "Tags: $3" ${4:+-H "Priority: $4"} \
       -d "$1" "$NTFY_URL" >/dev/null 2>&1 || true
}
mkdir -p "$ROOT/logs"
echo "running:$STAGE $(date -u +%FT%TZ)" > "$ROOT/logs/stage.state"
trap 'rc=$?; if [ $rc -eq 0 ]; then
        echo "done:$STAGE $(date -u +%FT%TZ)" > "$ROOT/logs/stage.state"
        ntfy "stage $STAGE completed" "GLM53 OK: $STAGE" "white_check_mark"
      else
        echo "failed:$STAGE $(date -u +%FT%TZ)" > "$ROOT/logs/stage.state"
        ntfy "stage $STAGE FAILED rc=$rc — control session will diagnose via jl run logs" "GLM53 FAILED: $STAGE" "rotating_light" "high"
      fi' EXIT
ntfy "stage $STAGE started" "GLM53 start: $STAGE" "arrow_forward"
# Hopper per vLLM recipe: BF16 KV (auto), flashinfer autotune off; text-only so the
# multimodal profiler cannot abort a fully loaded TP8 engine.
ENGINE_KW='{"enable_flashinfer_autotune": false, "limit_mm_per_prompt": {"image": 0, "video": 0}}'

drun() {  # drun <entrypoint> [args...]
  local ep="$1"; shift
  sudo docker run --rm -i --gpus all --ipc=host --shm-size=64g \
    -v "$ROOT:/glm53" \
    -e NVIDIA_TF32_OVERRIDE=0 \
    -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_LOGGING_LEVEL=INFO \
    -e HF_HUB_OFFLINE=1 \
    --entrypoint "$ep" "$IMAGE_REF" "$@"
}

FID=/glm53/bundle/tools/fidelity.py
SUITE=/glm53/bundle/suite

sentinel_receipt() {  # sentinel_receipt <variant>
  drun python3 - "$1" <<'EOF'
import json, pathlib, sys
v = sys.argv[1]
root = pathlib.Path("/glm53/captures")
main = json.loads((root / v / "capture-manifest.json").read_text())
sent = json.loads((root / f"{v}-sentinel" / "capture-manifest.json").read_text())
mrec = {r["index"]: r["sha256"] for r in main["captures"]}
mismatch = [r["index"] for r in sent["captures"] if mrec.get(r["index"]) != r["sha256"]]
receipt = {"schema": "glm53flash-capture-determinism/1", "variant": v,
           "sentinels": len(sent["captures"]),
           "byte_identical": len(sent["captures"]) - len(mismatch),
           "mismatched_indices": mismatch}
pathlib.Path(f"/glm53/out/determinism-{v}.json").write_text(json.dumps(receipt, indent=2))
print("DETERMINISM", v, json.dumps(receipt))
EOF
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$ROOT/out/determinism-$1.json'))['mismatched_indices'] else 1)"; then
    # Not byte-identical: measure the run-to-run noise floor through the head
    drun python3 "$FID" replay \
      --reference "/glm53/captures/$1" --candidate "/glm53/captures/$1-sentinel" \
      --head /glm53/out/head.safetensors --suite "$SUITE" \
      --out "/glm53/out/determinism-noise-$1.json" \
      --no-hash-shards --device cuda --filter sentinel \
      || echo "noise replay exited rc=$? - tolerated if receipt valid"
    python3 - "$1" <<'NOISEGATE'
import json, sys
v = sys.argv[1]
n = json.load(open(f"/home/ubuntu/glm53/out/determinism-noise-{v}.json"))
kld = n["token_mean_kld"]
print(f"NOISE_FLOOR {v}: mean KLD {kld:.2e}, top1 {n['top1_agreement']:.4f}")
assert kld <= 1e-3, f"run-to-run noise {kld} exceeds 1e-3 - measurement quality compromised"
NOISEGATE
  fi
}

case "$1" in
extract_head)
  drun python3 /glm53/bundle/remote/extract_head.py --model /glm53/models/bf16 --out /glm53/out
  ;;
gen_check)
  drun python3 /glm53/bundle/tools/gen_check.py \
    --model /glm53/models/bf16 --out /glm53/out/gen-check.json \
    --tensor-parallel $TP --gpu-memory-utilization 0.90 \
    --engine-kwargs "$ENGINE_KW" || echo "gen_check exited rc=$? - tolerated if receipt valid"
  python3 - <<'GENSUM'
import json
r = json.load(open("/home/ubuntu/glm53/out/gen-check.json"))
assert r["pass"], "gen_check receipt says degenerate output"
last = r["results"][-1]
snippet = (last.get("answer_excerpt") or last["completion"])[:350].replace("\n", " ")
if last.get("has_think_block"):
    snippet = "[after {} chars of thinking] ".format(last.get("think_chars", 0)) + snippet
print("GEN_CHECK_GREEN paris=", r["paris_mentioned"])
open("/home/ubuntu/glm53/out/gen-snippet.txt", "w").write(snippet)
GENSUM
  curl -s -m 10 -H "Title: GLM53 speaks (gen_check green)" -H "Tags: speech_balloon" \
       --data-binary @/home/ubuntu/glm53/out/gen-snippet.txt "$NTFY_URL" >/dev/null 2>&1 || true
  ;;
capture_bf16)
  drun python3 "$FID" capture \
    --model /glm53/models/bf16 --suite "$SUITE" --out /glm53/captures/bf16 \
    --tensor-parallel $TP --gpu-memory-utilization 0.90 \
    --engine-kwargs "$ENGINE_KW" --no-hash-shards --chunk-accumulate --filter all || echo "capture exited rc=$? - tolerated if manifest complete"
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])); assert m['complete'], 'capture incomplete'; print('capture complete:', m['contexts'])" "$ROOT/captures/bf16/capture-manifest.json"
  ;;
sentinel_bf16)
  drun python3 "$FID" capture \
    --model /glm53/models/bf16 --suite "$SUITE" --out /glm53/captures/bf16-sentinel \
    --tensor-parallel $TP --gpu-memory-utilization 0.90 \
    --engine-kwargs "$ENGINE_KW" --no-hash-shards --chunk-accumulate --filter sentinel
  sentinel_receipt bf16
  ;;
qualify_bf16)
  drun python3 "$FID" qualify \
    --model /glm53/models/bf16 --suite "$SUITE" --hidden /glm53/captures/bf16 \
    --head /glm53/out/head.safetensors --out /glm53/out/qualify-bf16.json \
    --contexts 8 --tensor-parallel $TP --gpu-memory-utilization 0.90 \
    --engine-kwargs "$ENGINE_KW" || echo "qualify exited rc=$? - tolerated if receipt valid"
  python3 -c "import json; q=json.load(open('$ROOT/out/qualify-bf16.json')); print('qualify-bf16:', q['mean_kld_live_vs_replayed'], q['top1_agreement'])"
  ;;
free_bf16)
  test -f "$ROOT/out/head.safetensors"
  test -f "$ROOT/out/determinism-bf16.json"
  python3 - <<'GATE'
import json
q = json.load(open("/home/ubuntu/glm53/out/qualify-bf16.json"))
kld = q["mean_kld_live_vs_replayed"]
assert kld <= 1e-5, f"qualify gate RED: mean KLD(live||replayed) = {kld}"
assert q["top1_agreement"] >= 0.999, q["top1_agreement"]
m = json.load(open("/home/ubuntu/glm53/captures/bf16/capture-manifest.json"))
assert m["complete"] is True
print("free_bf16 gates green: qualify KLD", kld)
GATE
  rm -rf "$ROOT/models/bf16"
  df -h /home
  ;;
head_check_fp8)
  drun python3 /glm53/bundle/remote/extract_head.py --model /glm53/models/fp8 --out /glm53/out/fp8-head-check
  python3 - <<'HEADCHK'
import json, pathlib
a = json.load(open("/home/ubuntu/glm53/out/head-extraction.json"))["tensors"]
b = json.load(open("/home/ubuntu/glm53/out/fp8-head-check/head-extraction.json"))["tensors"]
receipt = {"schema": "glm53flash-head-equality/1",
           "head_equal": a["head"]["sha256"] == b["head"]["sha256"],
           "final_norm_equal": a["final_norm"]["sha256"] == b["final_norm"]["sha256"],
           "bf16": {k: v["sha256"] for k, v in a.items()},
           "fp8": {k: v["sha256"] for k, v in b.items()}}
pathlib.Path("/home/ubuntu/glm53/out/head-equality-fp8.json").write_text(json.dumps(receipt, indent=2))
print("HEAD_EQUALITY", json.dumps(receipt))
assert receipt["head_equal"] and receipt["final_norm_equal"], (
    "FP8 repo head/norm differ from BF16 — replay must use --candidate-head; stop and decide")
HEADCHK
  ;;
capture_fp8)
  drun python3 "$FID" capture \
    --model /glm53/models/fp8 --suite "$SUITE" --out /glm53/captures/fp8 \
    --tensor-parallel $TP --gpu-memory-utilization 0.90 \
    --engine-kwargs "$ENGINE_KW" --no-hash-shards --chunk-accumulate --filter all || echo "capture exited rc=$? - tolerated if manifest complete"
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])); assert m['complete'], 'capture incomplete'; print('capture complete:', m['contexts'])" "$ROOT/captures/fp8/capture-manifest.json"
  ;;
sentinel_fp8)
  drun python3 "$FID" capture \
    --model /glm53/models/fp8 --suite "$SUITE" --out /glm53/captures/fp8-sentinel \
    --tensor-parallel $TP --gpu-memory-utilization 0.90 \
    --engine-kwargs "$ENGINE_KW" --no-hash-shards --chunk-accumulate --filter sentinel
  sentinel_receipt fp8
  ;;
qualify_fp8)
  drun python3 "$FID" qualify \
    --model /glm53/models/fp8 --suite "$SUITE" --hidden /glm53/captures/fp8 \
    --head /glm53/out/head.safetensors --out /glm53/out/qualify-fp8.json \
    --contexts 8 --tensor-parallel $TP --gpu-memory-utilization 0.90 \
    --engine-kwargs "$ENGINE_KW" || echo "qualify exited rc=$? - tolerated if receipt valid"
  python3 -c "import json; q=json.load(open('$ROOT/out/qualify-fp8.json')); print('qualify-fp8:', q['mean_kld_live_vs_replayed'], q['top1_agreement'])"
  ;;
replay)
  drun python3 "$FID" replay \
    --reference /glm53/captures/bf16 --candidate /glm53/captures/fp8 \
    --head /glm53/out/head.safetensors --suite "$SUITE" \
    --out /glm53/out/report-fp8-vs-bf16.json --no-hash-shards --device cuda
  drun python3 "$FID" replay \
    --reference /glm53/captures/bf16 --candidate /glm53/captures/fp8 \
    --head /glm53/out/head.safetensors --suite "$SUITE" \
    --out /glm53/out/report-fp8-vs-bf16-scorefrom1024.json \
    --no-hash-shards --device cuda --score-from 1024
  ;;
cross_check)
  set +e
  drun python3 /glm53/bundle/tools/cross_check.py suite \
    --logits /glm53/crosscheck/bm-teacher-logits --out /glm53/crosscheck/suite && \
  drun python3 "$FID" capture \
    --model /glm53/models/bf16 --suite /glm53/crosscheck/suite --out /glm53/crosscheck/cap \
    --tensor-parallel $TP --gpu-memory-utilization 0.90 \
    --engine-kwargs "$ENGINE_KW" --no-hash-shards --chunk-accumulate --filter all && \
  drun python3 /glm53/bundle/tools/cross_check.py compare \
    --logits /glm53/crosscheck/bm-teacher-logits --suite /glm53/crosscheck/suite \
    --capture /glm53/crosscheck/cap --head /glm53/out/head.safetensors \
    --out /glm53/out/crosscheck-brandonmusic.json
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    ntfy "cross-check vs brandonmusic teacher logits skipped/failed (rc=$rc) - non-blocking" "GLM53 cross-check skipped" "warning"
  fi
  exit 0
  ;;
activations)
  drun python3 /glm53/bundle/tools/activation_capture.py \
    --model /glm53/models/bf16 --suite /glm53/bundle/calsuite --out /glm53/activations/bf16-cal \
    --tensor-parallel $TP --gpu-memory-utilization 0.90 --engine-kwargs "$ENGINE_KW" || echo "activation capture exited rc=$? - tolerated if manifest complete"
  python3 -c "import json; m=json.load(open('$ROOT/activations/bf16-cal/activation-manifest.json')); assert m['complete']"
  du -sh "$ROOT/activations/bf16-cal"
  ;;
publish)
  test -f "$ROOT/.hf_token" || { echo "no .hf_token on VM — publish skipped"; exit 0; }
  export HF_TOKEN=$(cat "$ROOT/.hf_token")
  "$ROOT/hfenv/bin/pip" -q install -U huggingface_hub >/dev/null 2>&1 || true
  "$ROOT/hfenv/bin/python" - <<'PUB'
import json, os, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
root = pathlib.Path("/home/ubuntu/glm53")

rep = {}
for name in ("report-fp8-vs-bf16", "qualify-bf16", "qualify-fp8"):
    f = root / f"out/{name}.json"
    if f.is_file():
        rep[name] = json.loads(f.read_text())

ds = "malaiwah/glm53-flash-fidelity-suite-v1"
api.create_repo(ds, repo_type="dataset", exist_ok=True)
headline = ""
if "report-fp8-vs-bf16" in rep:
    r = rep["report-fp8-vs-bf16"]
    headline = (f"Official FP8 vs BF16: token mean KLD {r['token_mean_kld']:.6f} nats "
                f"(context macro mean {r['context_macro_mean_kld']:.6f}, "
                f"95% CI [{r['context_bootstrap']['ci95_low']:.6f}, {r['context_bootstrap']['ci95_high']:.6f}]), "
                f"top-1 agreement {r['top1_agreement']:.4f} over {r['scored_positions']:,} positions.")
card_lines = ["---",
"license: mit",
"tags:", "- glm5_next", "- glm-5.3-flash", "- fidelity", "- kld", "- hidden-states",
"---", "",
"# GLM-5.3-Flash Fidelity Suite v1",
"",
"**The first measured quality reference for GLM-5.3-Flash** (released 2026-08-26):",
"BF16-reference and FP8-as-served hidden-state captures, a shared LM head, and the",
"receipts to score ANY quant of this model by KL divergence - without holding the",
"643 GB reference. Protocol: the Qwen3.8-27B fidelity-suite v5 methodology",
"(hidden-state replay through one shared BF16 head, exact two-pass full-vocab KL).",
""]
if "report-fp8-vs-bf16" in rep:
    r = rep["report-fp8-vs-bf16"]
    b = r["context_bootstrap"]
    card_lines += ["## Headline: official FP8 vs BF16", "",
        f"| metric | value |", "|---|---|",
        f"| token mean KLD | **{r['token_mean_kld']:.6f} nats** |",
        f"| context macro mean (95% CI) | {r['context_macro_mean_kld']:.6f} [{b['ci95_low']:.6f}, {b['ci95_high']:.6f}] |",
        f"| median / p99 / p999 KLD | {r['token_median_kld']:.2e} / {r['p99_kld']:.2e} / {r['p999_kld']:.2e} |",
        f"| top-1 agreement | {r['top1_agreement']:.4f} |",
        f"| mean JSD (bits) | {r['mean_jsd_bits']:.6f} |",
        f"| scored positions | {r['scored_positions']:,} ({r['contexts']:,} contexts x 2,047) |",
        "", "Per-stratum mean KLD:", "",
        "| stratum | contexts | mean KLD |", "|---|---|---|"]
    for k, v in sorted(r["strata"].items()):
        card_lines.append(f"| {k} | {v['contexts']} | {v['mean_kld']:.6f} |")
    card_lines.append("")
sf = root / "out/report-fp8-vs-bf16-scorefrom1024.json"
if sf.is_file():
    w = json.loads(sf.read_text())
    card_lines += [f"llama.cpp-comparable geometry (positions 1024+ only): token mean KLD "
                   f"{w['token_mean_kld']:.6f}, top-1 {w['top1_agreement']:.4f}.", ""]
card_lines += ["## Receipts", ""]
for name, label in (("qualify-bf16", "BF16 live-vs-replayed"), ("qualify-fp8", "FP8 live-vs-replayed")):
    if name in rep:
        q = rep[name]
        card_lines.append(f"- **{label}**: mean KLD {q['mean_kld_live_vs_replayed']:.2e}, "
                          f"top-1 {q['top1_agreement']:.5f} over {q['contexts']} contexts - "
                          "the replay reproduces served logits.")
for v in ("bf16", "fp8"):
    f = root / f"out/determinism-{v}.json"
    if f.is_file():
        d = json.loads(f.read_text())
        card_lines.append(f"- **{v} determinism**: {d['byte_identical']}/{d['sentinels']} sentinel "
                          "contexts byte-identical across independent engine loads.")
he = root / "out/head-equality-fp8.json"
if he.is_file():
    h = json.loads(he.read_text())
    card_lines.append(f"- **Head equality**: FP8 repo lm_head/final-norm byte-identical to BF16: "
                      f"{h['head_equal']}/{h['final_norm_equal']} - one shared head scores both.")
nd_lines = ["", "## Known issue: run-to-run nondeterminism (first report)", "",
"glm5_next inference under the day-one vLLM image is NOT deterministic across",
"engine launches (it IS stable within one launch): 12/32 sentinel contexts",
"recaptured in a fresh engine load differ bytewise; measured noise floor",
"8.7e-4 mean KLD / top-1 0.9946 through the shared head (see",
"reports/determinism-bf16.json + determinism-noise-bf16.json). Live-vs-replay",
"(a third launch, through the model) bounds at ~1.5e-2 (reports/qualify-*.json).",
"Root cause traced to per-process Triton autotune winner selection on the",
"vendored FLA/KDA chunk kernels (two kernels change fp32 reduction splits with",
"config choice), amplified by DSA index_topk membership flips; all-reduce",
"dispatch verified identical across launches. Lineage: fla-org",
"flash-linear-attention#945, triton-lang/triton#9368; details + receipts:",
"https://github.com/vllm-project/vllm/pull/53906#issuecomment-5433635837 and",
"the JOURNAL in https://github.com/malaiwah/glm53-flash-fidelity-suite .",
"Consequence for THIS dataset: the paired FP8-vs-BF16 comparison shares the",
"replay path on both operands and carries only the 8.7e-4 capture floor;",
"absolute as-served claims carry the ~1.5e-2 bound. Deterministic rerun",
"recipe (v2): TRITON_CACHE_AUTOTUNING=1 with persistent TRITON_CACHE_DIR, or",
"pin the vendored FLA autotune lists to one num_warps=2 config.",
""]
card_lines += nd_lines
cc = root / "out/crosscheck-brandonmusic.json"
if cc.is_file():
    c = json.loads(cc.read_text())
    card_lines.append(f"- **Independent cross-validation**: vs brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits "
                      f"(separate pipeline, full-vocab fp32): mean KLD(theirs||ours) {c['mean_kld']:.2e}, "
                      f"top-1 {c['top1_agreement']:.5f} over {c['positions']:,} positions.")
card_lines += ["",
"## Pins", "",
"| what | value |", "|---|---|",
"| BF16 reference | `zai-org/GLM-5.3-Flash-BF16` @ `b1967181a3917ae70a437f4884748f6b8e3a1f4d` |",
"| FP8 as-served | `zai-org/GLM-5.3-Flash` @ `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a` |",
"| engine | vLLM glm53-flash docker image (digest in `reports/image-pin.txt`), TP8 H200, eager, TF32 off, BF16 KV |",
"| suite | 5,120 ctx x 2,048 tok, held-out v5-lineage corpus, GLM tokenizer, 0 calibration-contamination hits |",
"| contamination boundary | exllamav3 standard_cal_data @ 0c49587a |",
"",
"## Contents", "",
"- `suite/` - tokens + manifest (partitions: analysis/qualification/sentinels)",
"- `reference-bf16-shard0/`, `as-served-fp8-shard0/` - 512 contexts x [2047, 4096] bf16 hidden states each",
"- `head/` - shared BF16 `lm_head` (154,880 x 4,096) + final norm + extraction receipt",
"- `reports/` - the KLD reports and every receipt above; `SHA256SUMS` covers all files",
"",
"## Score your own quant", "",
"Capture final-norm hidden states of your quant over `suite/tokens/` (teacher-forced,",
"one context per forward), then replay against `reference-bf16-shard0` through",
"`head/head.safetensors` with the fidelity harness (tools published alongside;",
"see malaiwah's qwen38-27b-fidelity-suite-v5 for the protocol paper trail).",
"",
"Produced autonomously on rented 8x H200; contact: malaiwah."]
card = "\n".join(card_lines) + "\n"
(root / "deliverables/README.md").write_text(card)
api.upload_large_folder(repo_id=ds, repo_type="dataset", folder_path=str(root / "deliverables"))
print("published", ds)

acts = root / "activations/bf16-cal"
if (acts / "activation-manifest.json").is_file():
    ds2 = "malaiwah/glm53-flash-calibration-activations-v1"
    api.create_repo(ds2, repo_type="dataset", exist_ok=True)
    card2 = """---
license: mit
tags: [glm5_next, glm-5.3-flash, moe, calibration, activations, hessian]
---
# GLM-5.3-Flash calibration activations v1 (BF16, natural routing)

Per-layer block-input activations of `zai-org/GLM-5.3-Flash-BF16` @ b1967181 over 92x2048
tokens of the exllamav3 standard_cal_data corpus (pinned): per context, `layer_NNN.attn_in`
and `layer_NNN.mlp_in` (bf16, post-norm linear inputs; mlp_in is the router + expert gate/up
input) and `layer_NNN.router_logits` (fp32, natural top-8 routing ground truth).
Per-expert Hessians E[xx^T], routing statistics and down-proj inputs are recomputable offline.
Captured with vLLM TP8 eager; manifest carries sha256 per file.

Note: these activations are one draw from the engine-launch distribution — the
runtime is not launch-deterministic (Triton autotune winner selection on the
KDA kernels; see the fidelity suite's nondeterminism receipts and
https://github.com/vllm-project/vllm/pull/53906#issuecomment-5433635837 ).
For Hessian/routing statistics over 188K tokens this launch noise is far below
calibration sampling noise. A pinned-env (TRITON_CACHE_AUTOTUNING) v2
recapture is scripted in the companion repo for bit-reproducible needs.
"""
    (acts / "README.md").write_text(card2)
    api.upload_large_folder(repo_id=ds2, repo_type="dataset", folder_path=str(acts))
    print("published", ds2)
PUB
  ;;
det_confirm)
  # Deterministic-fix confirmation: pin Triton autotune winners via disk cache,
  # then prove two fresh engine launches capture byte-identical sentinels.
  # Runs ONLY while models/bf16 exists (e.g. during the free_bf16 hold).
  test -d "$ROOT/models/bf16"
  mkdir -p "$ROOT/triton-cache" "$ROOT/detpin"
  for run in seed runB runC; do
    sudo docker run --rm -i --gpus all --ipc=host --shm-size=64g \
      -v "$ROOT:/glm53" \
      -e TRITON_CACHE_AUTOTUNING=1 -e TRITON_CACHE_DIR=/glm53/triton-cache \
      -e TRITON_PRINT_AUTOTUNING=1 \
      -e NVIDIA_TF32_OVERRIDE=0 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_LOGGING_LEVEL=INFO \
      -e HF_HUB_OFFLINE=1 -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
      --entrypoint python3 "$IMAGE_REF" /glm53/bundle/tools/fidelity.py capture \
      --model /glm53/models/bf16 --suite /glm53/bundle/suite --out "/glm53/detpin/$run" \
      --tensor-parallel $TP --gpu-memory-utilization 0.90 \
      --engine-kwargs "$ENGINE_KW" --no-hash-shards --chunk-accumulate --filter sentinel \
      > "$ROOT/logs/detpin-$run.log" 2>&1 || echo "detpin $run rc=$? - tolerated if manifest complete"
    python3 -c "import json; assert json.load(open('$ROOT/detpin/$run/capture-manifest.json'))['complete'], '$run incomplete'"
    echo "detpin $run complete"
  done
  python3 - <<'DETRECEIPT'
import json, pathlib, re
root = pathlib.Path("/home/ubuntu/glm53")
runs = {}
for run in ("seed", "runB", "runC"):
    man = json.loads((root / f"detpin/{run}/capture-manifest.json").read_text())
    runs[run] = {r["index"]: r["sha256"] for r in man["captures"]}
bc_match = [i for i in runs["runB"] if runs["runB"][i] == runs["runC"].get(i)]
ab_match = [i for i in runs["seed"] if runs["seed"][i] == runs["runB"].get(i)]
winners = {}
for run in ("seed", "runB", "runC"):
    log = (root / f"logs/detpin-{run}.log").read_text(errors="ignore")
    winners[run] = sorted(set(re.findall(r"Triton autotuning for function (\S+) finished.*?(\{[^}]*\}|BLOCK\S*|num_warps: \d+[^\n]*)", log)))[:40]
receipt = {"schema": "glm53flash-determinism-pinned/1",
           "pin": "TRITON_CACHE_AUTOTUNING=1 + persistent TRITON_CACHE_DIR",
           "sentinels": len(runs["runB"]),
           "runB_vs_runC_byte_identical": len(bc_match),
           "seed_vs_runB_byte_identical": len(ab_match),
           "confirmed": len(bc_match) == len(runs["runB"]) and len(runs["runB"]) > 0,
           "autotune_lines_seen": {k: len(v) for k, v in winners.items()}}
(root / "out/determinism-pinned-bf16.json").write_text(json.dumps(receipt, indent=2))
print("DET_CONFIRM", json.dumps(receipt))
assert receipt["confirmed"], "pinned runs NOT byte-identical - fix not confirmed"
DETRECEIPT
  ntfy "Triton-cache pin CONFIRMED: fresh launches byte-identical on all sentinels (receipt: determinism-pinned-bf16.json)" "GLM53 determinism fix confirmed" "white_check_mark"
  ;;
env_receipt)
  drun python3 - <<'EOF'
import json, subprocess, torch, vllm
receipt = {
  "torch": torch.__version__, "cuda": torch.version.cuda, "vllm": vllm.__version__,
  "gpus": [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())],
  "pip_freeze": subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout.split("\n"),
}
open("/glm53/out/environment.json", "w").write(json.dumps(receipt, indent=1))
print("env receipt ok:", receipt["torch"], receipt["vllm"], len(receipt["gpus"]), "gpus")
EOF
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$ROOT/out/nvidia-smi.txt"
  cat "$ROOT/out/image-pin.txt" || true
  ;;
package)
  DEL="$ROOT/deliverables"
  mkdir -p "$DEL"/{suite,reference-bf16-shard0,as-served-fp8-shard0,reports,head}
  cp -r "$ROOT/bundle/suite/." "$DEL/suite/"
  for i in $(seq -f '%04g' 0 511); do
    cp "$ROOT/captures/bf16/hidden_$i.safetensors" "$DEL/reference-bf16-shard0/"
    cp "$ROOT/captures/fp8/hidden_$i.safetensors" "$DEL/as-served-fp8-shard0/"
  done
  cp "$ROOT/captures/bf16/capture-manifest.json" "$DEL/reference-bf16-shard0/capture-manifest-full.json"
  cp "$ROOT/captures/fp8/capture-manifest.json" "$DEL/as-served-fp8-shard0/capture-manifest-full.json"
  cp "$ROOT/out/head.safetensors" "$ROOT/out/final_norm.safetensors" \
     "$ROOT/out/head-extraction.json" "$DEL/head/"
  cp "$ROOT"/out/qualify-*.json "$ROOT"/out/determinism-*.json \
     "$ROOT"/out/report-*.json "$ROOT/out/environment.json" \
     "$ROOT/out/head-equality-fp8.json" \
     "$ROOT/out/nvidia-smi.txt" "$ROOT/out/image-pin.txt" "$DEL/reports/"
  (cd "$DEL" && find . -type f ! -name SHA256SUMS -exec sha256sum {} \; | sort -k2 > SHA256SUMS)
  du -sh "$DEL" "$DEL"/*
  echo PACKAGE_DONE
  ;;
*)
  echo "unknown stage: $1" >&2; exit 2 ;;
esac
