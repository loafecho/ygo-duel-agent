#!/bin/bash
# Pause training with the fake game, compare ZeroShot before/after flash-linear-attention, resume.
set -u
W=~/.local/share/ygo-worker; cd "$W" || exit 1
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv-cache YGO_DATA=$W/fla-check
mkdir -p $YGO_DATA/logs
$W/fake/gamescope -- 900 & FAKE=$!
for i in $(seq 1 60); do pgrep -f "app/[w]orker.py train" >/dev/null || break; sleep 1; done
echo "training paused after ${i}s; gpu: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) apps"
venv/bin/python -u app/fla_check.py $W/data/fla-prompts.json $YGO_DATA/before.json 2>&1 | grep -v "Loading\|Warning\|^\[transformers\]" | tail -2
uv pip install --python venv/bin/python flash-linear-attention==0.5.2 2>&1 | tail -4
venv/bin/python -u app/fla_check.py $W/data/fla-prompts.json $YGO_DATA/after.json 2>&1 | grep -v "Loading\|Warning" | tail -3
venv/bin/python - <<PY
import json
a=json.load(open("$YGO_DATA/before.json")); b=json.load(open("$YGO_DATA/after.json"))
same=sum(x==y for x,y in zip(a["choices"],b["choices"]))
gap=max(abs(u-v) for la,lb in zip(a["logits"],b["logits"]) for u,v in zip(la,lb))
print(f"fla_check: same_choices={same}/{len(a['choices'])} max_logit_diff={gap:.3f} before_s={a['seconds']} after_s={b['seconds']} speedup={a['seconds']/b['seconds']:.2f}x")
PY
uv cache clean >/dev/null 2>&1
kill $FAKE; echo "fake game ended; training resumes"
