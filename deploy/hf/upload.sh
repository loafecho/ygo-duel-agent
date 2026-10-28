#!/usr/bin/env bash
# Publish the round-5 adapter to Hugging Face. Run on the GPU workstation: bash upload.sh <hf-user>/<repo-name>
# The token comes from HF_TOKEN or the 0600 file $W/hf/token (huggingface_hub reads it via HF_HOME),
# never from an argument or a file in this repo, and is never printed.
set -euo pipefail
REPO=${1:?usage: upload.sh <hf-user>/<repo-name>}
W=~/.local/share/ygo-worker
D=$W/hf-export/yugioh-kashtira-qwen2.5-1.5b-lora
export HF_HOME=$W/hf
[ -n "${HF_TOKEN+SET}" ] || [ -s "$HF_HOME/token" ] || { echo "no token: set HF_TOKEN or write $HF_HOME/token"; exit 2; }
cp "$W/app/hf-README.md" "$D/README.md"
"$W/venv/bin/python" - "$REPO" "$D" <<'PY'
import sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1], sys.argv[2]
api = HfApi()
print("as", api.whoami()["name"])
api.create_repo(repo, repo_type="model", exist_ok=True)
info = api.upload_folder(repo_id=repo, folder_path=folder, commit_message="Round 5 Kashtira adapter (0.936 vs WindBot)")
files = sorted(f.rfilename for f in api.model_info(repo, files_metadata=True).siblings)
print("uploaded", info.commit_url)
print("files", files)
PY
