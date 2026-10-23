#!/bin/bash
# Round 3 on the GPU workstation: imitation first, then a polish vs WindBot. Needs $W/labels-r3.jsonl
# (tools/build_labels.py on the control host: WindBot's picks + the teacher's, WindBot held-out games marked).
#   textcheck  step-1239 on held-out WindBot decisions, with and without card text -> data-r3/done-textcheck = full|oneliner
#   pause      pause_test.sh against the round-3 trainer (own pause-data dir)       -> logs/pause-test-r3.txt
#   train      supervised on the labels, then PPO vs WindBot; 12 GPU-hours, 24 if still climbing, stop at >= 0.50
#   table      the final checkpoint and up to 4 earlier ones, 100 games vs WindBot and vs step-1239
# Every step runs under supervisor.py (the game guard). A finished step leaves stage-r3-<name>.done.
# Log: logs/pipeline-r3.log
set -u
W=~/.local/share/ygo-worker
export HF_HOME=$W/hf
LOG=$W/logs/pipeline-r3.log
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
sup() {  # sup <mode>: run one supervised step in data-r3
  YGO_DATA=$W/data-r3 /usr/bin/python3 -I "$W/app/supervisor.py" "$1" >> "$W/logs/supervisor-$1-r3.out" 2>&1 < /dev/null
  [ -e "$W/data-r3/done-$1" ]
}
export YGO_LABELS_FILE=$W/labels-r3.jsonl YGO_WARM=$W/data/ckpt/pool/step-1239.pt
mkdir -p "$W/data-r3/logs"

say "pipeline-r3 up: $(free_gb) GB free, labels $(wc -l < "$YGO_LABELS_FILE")"
if [ ! -e "$W/stage-r3-textcheck.done" ]; then
  wait_idle
  say "textcheck: start"
  if sup textcheck; then touch "$W/stage-r3-textcheck.done"; say "textcheck: $(cat "$W/data-r3/done-textcheck")"
  else say "textcheck: supervisor exited without done-textcheck"; exit 1; fi
fi
YGO_CARD_TEXT=$(cat "$W/data-r3/done-textcheck"); export YGO_CARD_TEXT

if [ ! -e "$W/stage-r3-pause.done" ]; then
  for attempt in 1 2 3; do
    wait_idle
    say "pause: attempt $attempt (card_text=$YGO_CARD_TEXT)"
    if bash "$W/app/pause_test.sh" > "$W/logs/pause-test-r3.txt" 2>&1; then touch "$W/stage-r3-pause.done"; say "pause: done"; break; fi
    say "pause: failed (logs/pause-test-r3.txt)"
  done
fi

export YGO_CAP_H=12 YGO_CAP2_H=24 YGO_EVAL_EVERY_H=2 YGO_EVAL_OPPS=windbot YGO_POOL_EVERY_S=3600 YGO_RECIPE=3 \
  YGO_RECIPE_CHANGE='{"windbot_share": 0.7, "pool_share": 0.15, "random_share": 0.0}'
if [ ! -e "$W/stage-r3-train.done" ]; then
  say "train: supervisor start (card_text=$YGO_CARD_TEXT)"
  if sup train; then touch "$W/stage-r3-train.done"; say "train: done ($(cat "$W/data-r3/done-train"))"
  else say "train: supervisor exited without done-train"; exit 1; fi
fi

if [ ! -e "$W/stage-r3-table.done" ]; then
  say "table: start"
  if YGO_TABLE_OPPS=windbot,ref YGO_TABLE_REF=$YGO_WARM sup table; then touch "$W/stage-r3-table.done"; say "table: done"
  else say "table: supervisor exited without done-table"; exit 1; fi
fi
say "pipeline-r3 done: $(free_gb) GB free"
