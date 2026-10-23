#!/bin/bash
# Round 2 on the GPU workstation: card-text prompts, WindBot opponent, loop penalty, warm start.
#   pause  pause_test.sh against the round-2 trainer (own pause-data dir)    -> logs/pause-test-r2.txt
#   train  supervisor.py train in data-r2: warm start from step-1239, until vs WindBot >= 0.50 or 24 GPU-hours
# The optional 12-hour second run (one recipe change) and the table are started by hand after reading the result.
# A finished step leaves stage-r2-<name>.done. Log: logs/pipeline-r2.log
set -u
W=~/.local/share/ygo-worker
export HF_HOME=$W/hf
LOG=$W/logs/pipeline-r2.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
game() { /usr/bin/python3 -I "$W/app/supervisor.py" check 2>/dev/null || echo "check-failed"; }
free_gb() { sync; df -BG --output=avail /var/home | tail -1 | tr -dc 0-9; }
wait_idle() {
  local n=0
  while true; do
    if [ "$(game)" = idle ]; then n=$((n + 1)); [ $n -ge 3 ] && return; else n=0; fi
    sleep 10
  done
}
export YGO_WARM=$W/data/ckpt/pool/step-1239.pt YGO_POOL_EVERY_S=3600

say "pipeline-r2 up: $(free_gb) GB free"
if [ ! -e "$W/stage-r2-pause.done" ]; then
  for attempt in 1 2 3; do
    wait_idle
    say "pause: attempt $attempt"
    if bash "$W/app/pause_test.sh" > "$W/logs/pause-test-r2.txt" 2>&1; then touch "$W/stage-r2-pause.done"; say "pause: done"; break; fi
    say "pause: failed (logs/pause-test-r2.txt)"
  done
fi
if [ ! -e "$W/stage-r2-train.done" ]; then
  say "train: supervisor start"
  YGO_DATA=$W/data-r2 /usr/bin/python3 -I "$W/app/supervisor.py" train >> "$W/logs/supervisor-train-r2.out" 2>&1 < /dev/null
  if [ -e "$W/data-r2/done-train" ]; then touch "$W/stage-r2-train.done"; say "train: done ($(cat "$W/data-r2/done-train"))"
  else say "train: supervisor exited without done-train"; exit 1; fi
fi
say "pipeline-r2 done: $(free_gb) GB free"
