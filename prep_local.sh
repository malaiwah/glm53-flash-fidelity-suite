#!/bin/bash
# Local prep for the GLM-5.3-Flash fidelity suite: venv + corpus + cal data + tokenizer.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
./.venv/bin/pip -q install --upgrade pip
./.venv/bin/pip -q install "transformers>=4.55" "huggingface_hub[hf_transfer]" tokenizers

export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== v5 corpus (archival, byte-identical) ==="
./.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="malaiwah/qwen38-27b-fidelity-suite-v5", repo_type="dataset",
                      allow_patterns=["corpus/*"], local_dir="corpus_dl")
print("corpus ->", p)
EOF

echo "=== exllamav3 standard_cal_data (contamination boundary) ==="
./.venv/bin/python - <<'EOF'
import json, urllib.request, pathlib
api = "https://api.github.com/repos/turboderp-org/exllamav3"
head = json.load(urllib.request.urlopen(api + "/commits/master"))["sha"]
listing = json.load(urllib.request.urlopen(
    f"{api}/contents/exllamav3/conversion/standard_cal_data?ref={head}"))
out = pathlib.Path("cal_data"); out.mkdir(exist_ok=True)
for item in listing:
    if item["name"].endswith(".utf8"):
        urllib.request.urlretrieve(item["download_url"], out / item["name"])
        print("cal", item["name"], item["size"])
(out / "SOURCE.json").write_text(json.dumps(
    {"repo": "turboderp-org/exllamav3", "commit": head,
     "path": "exllamav3/conversion/standard_cal_data"}, indent=1))
print("pinned exllamav3 commit", head)
EOF

echo "=== GLM-5.3-Flash tokenizer snapshot (BF16 repo, pinned) ==="
./.venv/bin/python - <<'EOF'
import json, pathlib
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
for repo in ("zai-org/GLM-5.3-Flash", "zai-org/GLM-5.3-Flash-BF16"):
    info = api.model_info(repo)
    print(repo, "revision", info.sha, "lastModified", info.last_modified)
info = api.model_info("zai-org/GLM-5.3-Flash-BF16")
p = snapshot_download(repo_id="zai-org/GLM-5.3-Flash-BF16", revision=info.sha,
                      allow_patterns=["tokenizer*", "*config.json", "chat_template*"],
                      local_dir="tokenizer")
pathlib.Path("tokenizer/revision.txt").write_text(info.sha + "\n")
print("tokenizer ->", p, "| pinned", info.sha)
EOF

echo "=== summary ==="
du -sh corpus_dl/corpus/text cal_data tokenizer 2>/dev/null
ls corpus_dl/corpus/text | head -3
echo PREP_DONE
