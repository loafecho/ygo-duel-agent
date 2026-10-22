#!/bin/bash
# Phases 1-3 on the GPU workstation, in order. Every step waits for no game (supervisor.py check) and is
# stopped when one starts; the supervisor stages pause and resume on their own.
#   upgrade  torch 2.14.1+cu130 / transformers 5.20.2 / peft 0.21.2 in the existing venv (no second venv)
#   download Qwen3.5-4B into hf/ (checks the 10 GB free-space floor first)
#   pause    pause_test.sh against the new trainer   -> logs/pause-test.txt
#   zeroshot    supervisor.py zeroshot (200 games vs random)
#   train    supervisor.py train (until eval >= 0.60 or the GPU-hour cap)
#   table    supervisor.py table
# A finished step leaves stage-<name>.done; rerunning the script skips it. Log: logs/pipeline.log
set -u
W=~/.local/share/ygo-worker
export UV_CACHE_DIR=$W/uv-cache UV_PYTHON_INSTALL_DIR=$W/python HF_HOME=$W/hf
LOG=$W/logs/pipeline.log
FLOOR_GB=10
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
game() { /usr/bin/python3 -I "$W/app/supervisor.py" check 2>/dev/null || echo "check-failed"; }
free_gb() { sync; df -BG --output=avail /var/home | tail -1 | tr -dc 0-9; }
wait_idle() {
  local n=0 r
  while true; do
    r=$(game)
    if [ "$r" = idle ]; then n=$((n + 1)); [ $n -ge 3 ] && return; else n=0; fi
    sleep 10
  done
}
# guarded NAME CMD...: run CMD until it exits 0, stopping it while a game runs and rerunning after.
guarded() {
  local name=$1 pid r rc stopped attempt fails=0
  shift
  [ -e "$W/stage-$name.done" ] && { say "$name: already done"; return 0; }
  for attempt in $(seq 1 30); do
    wait_idle
    say "$name: start (attempt $attempt, $(free_gb) GB free)"
    setsid bash -c "$*" >> "$W/logs/$name.out" 2>&1 < /dev/null &  # "$*" names an exported function
    pid=$!
    stopped=""
    while kill -0 $pid 2>/dev/null; do
      r=$(game)
      if [ "$r" != idle ] && [ "$name" != pause ]; then  # the pause test makes its own fake game
        say "$name: game ($r), stopping"
        kill -TERM -- -$pid 2>/dev/null
        stopped=1
        break
      fi
      sleep 5
    done
    wait $pid; rc=$?
    if [ -z "$stopped" ] && [ $rc -eq 0 ]; then
      touch "$W/stage-$name.done"; say "$name: done"; return 0
    fi
    if [ -z "$stopped" ]; then
      fails=$((fails + 1)); say "$name: failed rc=$rc (see logs/$name.out)"
      [ $fails -ge 3 ] && return 1
    fi
  done
  return 1
}
# supervised NAME: the supervisor handles games itself; done when it leaves data/done-NAME.
supervised() {
  local name=$1
  [ -e "$W/stage-$name.done" ] && { say "$name: already done"; return 0; }
  say "$name: supervisor start"
  /usr/bin/python3 -I "$W/app/supervisor.py" "$name" >> "$W/logs/supervisor-$name.out" 2>&1 < /dev/null
  if [ -e "$W/data/done-$name" ]; then touch "$W/stage-$name.done"; say "$name: done ($(cat "$W/data/done-$name"))"; return 0; fi
  say "$name: supervisor exited without done-$name"; return 1
}

upgrade() {
  cd "$W" &&
  uv pip install --python venv/bin/python --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1 &&
  uv pip install --python venv/bin/python transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 &&
  venv/bin/python -c "import torch,transformers,peft;print('cuda',torch.cuda.is_available(),torch.__version__,transformers.__version__,peft.__version__)" &&
  uv cache clean
}
download() {
  local f have; f=$(free_gb)
  have=$(du -s -BG "$HF_HOME/hub/models--Qwen--Qwen3.5-4B" 2>/dev/null | tr -dc 0-9); have=${have:-0}
  if [ "$f" -lt $((FLOOR_GB + 9 - have)) ]; then echo "free ${f} GB, ${have} GB fetched: the rest of Qwen3.5-4B (~9 GB) would break the ${FLOOR_GB} GB floor"; return 1; fi
  "$W/venv/bin/python" -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-4B', allow_patterns=['*.json','*.safetensors','*.jinja','*.txt']))"
}
pause() { bash "$W/app/pause_test.sh" > "$W/logs/pause-test.txt" 2>&1; cat "$W/logs/pause-test.txt"; }

export W FLOOR_GB
export -f say free_gb upgrade download pause
say "pipeline up: $(free_gb) GB free"
guarded upgrade upgrade || { say "STOP upgrade"; exit 1; }
guarded download download || { say "STOP download"; exit 1; }
say "free after downloads: $(free_gb) GB"
guarded pause pause || { say "STOP pause"; exit 1; }
YGO_MAX_CRASHES=3 supervised zeroshot || { say "STOP zeroshot"; exit 1; }
# the Phase 0 smoke trainer left a checkpoint in another format: keep it, out of the new trainer's way
if [ -e "$W/data/ckpt/state.json" ] && ! grep -q gpu_s "$W/data/ckpt/state.json"; then mv "$W/data/ckpt" "$W/data/phase0-ckpt"; fi
supervised train || { say "STOP train"; exit 1; }
supervised table || { say "STOP table"; exit 1; }
say "pipeline done"
