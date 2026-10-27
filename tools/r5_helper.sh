#!/bin/bash
# Round 5 control-host helper: builds the DAgger label sets the GPU-host pipeline waits for, as soon as each shadow
# collection finishes (the shadow labels live on the game host, the collected games' seeds on the GPU workstation). Log: runs/r5/helper.log
set -u
R=${YGO_REPO:-$HOME/ygo-duel-agent}
O=$R/runs/r5
W=.local/share/ygo-worker
DUEL=${YGO_GAME_HOST_EXEC:-ssh -o BatchMode=yes game-host}  # command prefix that runs on the game host
GPU=${YGO_GPU_HOST:-gpu-host}  # ssh name of the GPU workstation
log() { echo "$(date '+%F %T') $*" >> "$O/helper.log"; }
dagger() {  # dagger <collect-name> <base labels on the GPU workstation> <out name> <src>
  local m=$1 base=$2 out=$3 src=$4
  ssh "$GPU" "test -e $W/$out" && return 0
  ssh "$GPU" "test -e $W/data-r5/m-$m/done-match" || return 1
  log "$m done: building $out from $base"
  scp -q "$GPU:$W/data-r5/m-$m/match-state.json" "$O/$m-state.json" || return 1
  $DUEL cat /srv/ygo/logs/dagger-labels.jsonl > "$O/dagger-labels.jsonl" || return 1
  scp -q "$GPU:$base" "$O/base-$out" || return 1
  python3 "$R/tools/build_dagger.py" --base "$O/base-$out" --dagger "$O/dagger-labels.jsonl" --state "$O/$m-state.json" \
    --src "$src" --out "$O/$out" --base-ratio 1 >> "$O/helper.log" 2>&1 || return 1
  scp -q "$O/$out" "$GPU:$W/$out.tmp" && ssh "$GPU" "mv $W/$out.tmp $W/$out" && log "$out delivered"
  rm -f "$O/base-$out" "$O/dagger-labels.jsonl"
}
log "helper up"
while true; do
  dagger deck-dagger-collect "$W/labels-r5deck.jsonl" labels-r5deck-dagger.jsonl dagger
  # round 5: dagger2 labels wait for runs/r5/swap.ok (the pipeline is relaunched at that wait with a resized trial)
  if [ -e "$O/swap.ok" ] && ssh "$GPU" "test -e $W/data-r5/m-dagger2-collect/done-match"; then
    base=$(ssh "$GPU" "cat $W/data-r5/labels")
    dagger dagger2-collect "$base" labels-dagger2.jsonl dagger2 && ssh "$GPU" "test -e $W/labels-dagger2.jsonl" && { log "helper done"; exit 0; }
  fi
  sleep 60
done
