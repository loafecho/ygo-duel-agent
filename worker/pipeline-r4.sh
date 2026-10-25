#!/bin/bash
# Round 4 on the GPU workstation: a pre-registered baseline, then prompt fixes, a gated critic, gated
# shadow-WindBot corrections (DAgger), practice with the critic, a pre-registered final, and search only on success.
# Every GPU step runs under supervisor.py (the game guard). A finished stage leaves stage-r4-<name>.done.
# Every eval is greedy, half the games as first player, draws count as losses, Wilson 95% intervals.
# GPU-h ledger: data-r4/ledger.jsonl, one line per finished stage; 24 GPU-h cap, 2 reserved for the final.
# Log: logs/pipeline-r4.log
set -u
W=~/.local/share/ygo-worker
D=$W/data-r4
export HF_HOME=$W/hf YGO_CARD_TEXT=${YGO_CARD_TEXT:-oneliner}
LOG=$W/logs/pipeline-r4.log
R3=$W/data-r3/ckpt/pool
CAP_H=24 RESERVE_H=2
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
py() { /usr/bin/python3 -I -c "$@"; }
ledger_h() { py 'import json,sys; print(round(sum(json.loads(l)["gpu_s"] for l in open(sys.argv[1])) / 3600, 3))' "$D/ledger.jsonl"; }
ledger_add() { echo "{\"stage\": \"$1\", \"gpu_s\": $2}" >> "$D/ledger.jsonl"; say "ledger: $1 +$(py "print(round($2/3600,3))") GPU-h, total $(ledger_h)"; }
left_h() { py "print(round($CAP_H - $RESERVE_H - $(ledger_h), 3))"; }  # GPU-h left before the final's reserve
field() { py 'import json,sys; v=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, v))' "$1" "$2"; }

match() {  # match <name> <A.pt> <B: windbot | .pt> <games> <seed>: one supervised match in $D/m-<name>
  local dir=$D/m-$1
  if [ ! -e "$dir/done-match" ]; then
    wait_idle
    say "match $1: start ($(basename "$2") vs $(basename "$3"), $4 games)"
    mkdir -p "$dir/logs"
    YGO_DATA=$dir YGO_MATCH_NAME=r4-$1 YGO_MATCH_A=$2 YGO_MATCH_B=$3 YGO_MATCH_GAMES=$4 YGO_MATCH_SEED=$5 \
      /usr/bin/python3 -I "$W/app/supervisor.py" match >> "$W/logs/supervisor-match-r4.out" 2>&1 < /dev/null
    [ -e "$dir/done-match" ] || { say "match $1: supervisor exited without done-match"; return 1; }
    ledger_add "match-$1" "$(field "$dir/result.json" gpu_s)"
  fi
}
ci() { field "$1" "f'{win_rate:.3f} ci95=[{ci95[0]:.3f},{ci95[1]:.3f}] games={games}'"; }

mkdir -p "$D/logs"
touch "$D/ledger.jsonl"
say "pipeline-r4 up: $(free_gb) GB free, ledger $(ledger_h) GPU-h, card_text=$YGO_CARD_TEXT"

# 1. Baseline, pre-registered, on the round-3 runner (before any runner change).
if [ ! -e "$W/stage-r4-baseline.done" ]; then
  match baseline-wb "$R3/step-10003.pt" windbot 500 41001 || exit 1
  match baseline-h2h "$R3/step-10003.pt" "$R3/step-6586.pt" 400 41002 || exit 1
  say "baseline vs_windbot=$(ci "$D/m-baseline-wb/result.json")"
  say "h2h step10003_vs_step6586=$(ci "$D/m-baseline-h2h/result.json")"
  ext=$(field "$D/m-baseline-h2h/result.json" "'yes' if win_rate >= 0.58 and ci95[0] > 0.50 else 'no'")
  echo "$ext" > "$D/extension-needed"
  say "baseline: unchanged-recipe extension needed=$ext (h2h >= 0.58 with lo > 0.50)"
  touch "$W/stage-r4-baseline.done"
fi

sup() {  # sup <mode> <dir>: one supervised run of a worker mode in its own data dir (env passed through)
  wait_idle
  mkdir -p "$2/logs"
  YGO_DATA=$2 /usr/bin/python3 -I "$W/app/supervisor.py" "$1" >> "$W/logs/supervisor-$1-r4.out" 2>&1 < /dev/null
  [ -e "$2/done-$1" ]
}
wait_file() {  # a control-host or game-host step (runner deploy, labels) leaves this file
  [ -e "$1" ] || say "waiting for $1 ($2)"
  until [ -e "$1" ]; do sleep 30; done
}
setv() { echo "$2" > "$D/$1"; }
getv() { cat "$D/$1"; }
max() { py "print(max($1, $2))"; }

# 2. Stable options: the round-4 runner is deployed on the game host (CPU checks run there).
wait_file "$W/stage-r4-runner.done" "options_stable=500/500 and random games=100 errors=0 on the deployed runner"
grep -h 'options_stable\|games=100 errors' "$W/stage-r4-runner.done" | while read -r l; do say "runner: $l"; done

# 3. Prompt fixes: history + checked one-liners + the leak check (CPU, game host), then a short re-imitation.
wait_file "$W/stage-r4-prompts.done" "oneliners and leakcheck lines, labels-r4.jsonl copied"
grep -h 'oneliners cards\|leakcheck prompts' "$W/stage-r4-prompts.done" | while read -r l; do say "prompts: $l"; done
BASE=$(field "$D/m-baseline-wb/result.json" win_rate)
export SL_LR_R4=3e-5
if [ ! -e "$W/stage-r4-reimitate.done" ]; then
  if [ ! -e "$D/sl-prompt/done-sl" ]; then
    say "re-imitation: start (step-10003, labels-r4, lr $SL_LR_R4)"
    YGO_WARM=$R3/step-10003.pt YGO_LABELS_FILE=$W/labels-r4.jsonl YGO_SL_LR=$SL_LR_R4 YGO_ONELINERS=oneliners-r4.json \
      YGO_HISTORY=1 sup sl "$D/sl-prompt" || { say "re-imitation: supervisor exited without done-sl"; exit 1; }
    ledger_add sl-prompt "$(field "$D/sl-prompt/result.json" gpu_s)"
    say "re-imitation: heldout_acc=$(field "$D/sl-prompt/result.json" heldout_acc)"
  fi
  YGO_ONELINERS=oneliners-r4.json YGO_HISTORY=1 YGO_MATCH_VALUE_LOG=$D/value-model.jsonl \
    match prompt-eval "$D/sl-prompt/final.pt" windbot 500 43001 || exit 1
  PX=$(field "$D/m-prompt-eval/result.json" win_rate)
  say "prompt_eval vs_windbot=$(ci "$D/m-prompt-eval/result.json") (baseline $BASE)"
  if py "import sys; sys.exit(0 if $PX < $BASE - 0.03 else 1)"; then
    say "prompt_fix=reverted ($PX < $BASE - 0.03): round-3 prompts and step-10003"
    setv fmt r3; setv current "$R3/step-10003.pt"; setv current_wr "$BASE"
  else
    say "prompt_fix=kept ($PX >= $BASE - 0.03)"
    setv fmt r4; setv current "$D/sl-prompt/final.pt"; setv current_wr "$PX"
  fi
  setv best_wr "$(max "$BASE" "$PX")"
  touch "$W/stage-r4-reimitate.done"
fi
if [ "$(getv fmt)" = r3 ]; then export YGO_HISTORY=0 YGO_ONELINERS=oneliners.json; else export YGO_HISTORY=1 YGO_ONELINERS=oneliners-r4.json; fi
say "current $(getv current) fmt=$(getv fmt) current_wr=$(getv current_wr) best_wr=$(getv best_wr)"

# 4. Board value (critic), gated.
if [ ! -e "$W/stage-r4-values.done" ]; then
  YGO_WARM=$(getv current) YGO_VALUE_FILES=$W/value-r4.jsonl,$D/value-model.jsonl sup values "$D/values" \
    || { say "values: supervisor exited without done-values"; exit 1; }
  ledger_add values "$(field "$D/values/result.json" gpu_s)"
  say "$(field "$D/values/result.json" "f'value auc={auc} brier={brier} brier_lp={brier_lp} games={games}'")"
  say "critic=$(field "$D/values/result.json" critic) (on iff auc >= 0.75 and brier <= 0.9 x brier_lp)"
  touch "$W/stage-r4-values.done"
fi
CRITIC=$(field "$D/values/result.json" critic)

# 5. Shadow-WindBot DAgger, gated.
if [ ! -e "$W/stage-r4-dagger.done" ]; then
  YGO_MATCH_SHADOW=1 match dagger-collect "$(getv current)" windbot 300 45001 || exit 1
  R=$D/m-dagger-collect/result.json
  say "dagger games=$(field "$R" games) states=$(field "$R" 'shadow[1]') coverage=$(field "$R" coverage) disagree=$(field "$R" disagree) (asked $(field "$R" 'shadow[0]'))"
  if py "import sys; sys.exit(0 if $(field "$R" coverage) < 0.80 else 1)"; then
    say "dagger=skipped (coverage < 0.80)"; setv dagger skipped
  else
    wait_file "$W/labels-dagger.jsonl" "DAgger labels from the game host aggregated with the WindBot set"
    if [ ! -e "$D/sl-dagger/done-sl" ]; then
      YGO_WARM=$(getv current) YGO_LABELS_FILE=$W/labels-dagger.jsonl YGO_SL_LR=$SL_LR_R4 sup sl "$D/sl-dagger" \
        || { say "dagger sl: supervisor exited without done-sl"; exit 1; }
      ledger_add sl-dagger "$(field "$D/sl-dagger/result.json" gpu_s)"
    fi
    match dagger-eval "$D/sl-dagger/final.pt" windbot 500 46001 || exit 1
    DX=$(field "$D/m-dagger-eval/result.json" win_rate)
    say "dagger_eval vs_windbot=$(ci "$D/m-dagger-eval/result.json") (best so far $(getv best_wr))"
    if py "import sys; sys.exit(0 if $DX > $(getv best_wr) else 1)"; then
      say "dagger=kept"; setv dagger kept; setv current "$D/sl-dagger/final.pt"; setv current_wr "$DX"; setv best_wr "$DX"
    else
      say "dagger=reverted ($DX <= best $(getv best_wr))"; setv dagger reverted
    fi
  fi
  touch "$W/stage-r4-dagger.done"
fi

# 6. Practice with the critic (<= 10 GPU-h, inside the cap with the final's 2 GPU-h kept back).
if [ ! -e "$W/stage-r4-practice.done" ]; then
  [ -e "$D/practice-cap" ] || setv practice-cap "$(py "print(round(max(0.0, min(10.0, $(left_h) - 0.3)), 2))")"
  say "practice: cap $(getv practice-cap) GPU-h, critic=$CRITIC, start $(getv current) at $(getv current_wr)"
  VH=""; [ "$CRITIC" = on ] && VH=$D/values/vhead.pt
  YGO_WARM=$(getv current) YGO_VHEAD=$VH YGO_CAP_H=$(getv practice-cap) YGO_START_WR=$(getv current_wr) \
    sup practice "$D/practice" || { say "practice: supervisor exited without done-practice"; exit 1; }
  ledger_add practice "$(field "$D/practice/result.json" gpu_s)"
  py 'import json,sys; [print(f"eval step={e[\"step\"]} gpu_h={e[\"gpu_h\"]} vs_windbot={e[\"vs_windbot\"]:.3f} games={e[\"games\"]} clipfrac={e[\"clipfrac\"]}") for e in json.load(open(sys.argv[1]))["evals"]]' \
    "$D/practice/result.json" | while read -r l; do say "$l"; done
  say "train_stop reason=$(cat "$D/practice/done-practice")"
  say "practice best: $(field "$D/practice/result.json" best)"
  touch "$W/stage-r4-practice.done"
fi

# 7. Final, pre-registered: the chosen checkpoint, 500 games vs WindBot and 300 paired games vs step-10003.
if [ ! -e "$W/stage-r4-final.done" ]; then
  FIN=$(field "$D/practice/result.json" "best['ckpt']")
  STEP=$(field "$D/practice/result.json" "best['step']")
  match final-wb "$FIN" windbot 500 47001 || exit 1
  YGO_MATCH_PAIRED=1 YGO_MATCH_B_FMT=r3 match final-r3 "$FIN" "$R3/step-10003.pt" 300 47002 || exit 1
  A=$D/m-final-wb/result.json B=$D/m-final-r3/result.json
  ERR=$(py "print($(field "$A" errors) + $(field "$B" errors))")
  say "final step=$STEP vs_windbot=$(field "$A" "f'{win_rate:.3f} ci95=[{ci95[0]:.3f},{ci95[1]:.3f}] games={games}'") vs_r3=$(ci "$B") errors=$ERR"
  V=$(py "print('success' if $(field "$A" win_rate) >= 0.55 and $(field "$A" 'ci95[0]') > 0.50 and $(field "$B" 'ci95[0]') > 0.50 and $ERR == 0 else 'miss')")
  say "verdict=$V"; setv verdict "$V"
  touch "$W/stage-r4-final.done"
fi

# 8. Search, only on success.
if [ ! -e "$W/stage-r4-search.done" ]; then
  if [ "$(getv verdict)" != success ]; then
    say "search skipped reason=miss (verdict=$(getv verdict))"
    touch "$W/stage-r4-search.done"
  else
    wait_file "$W/stage-r4-search.done" "search stage (built only on success)"
  fi
fi

# 9. Pause test against the round-4 trainer.
if [ ! -e "$W/stage-r4-pause.done" ]; then
  for attempt in 1 2 3; do
    wait_idle
    say "pause: attempt $attempt (round-4 trainer: practice)"
    if YGO_PAUSE_MODE=practice YGO_PAUSE_DATA=$W/pause-data-r4 YGO_WARM=$(getv current) YGO_START_WR=$(getv current_wr) \
       YGO_COLLECT=256 YGO_MINI=64 YGO_EVAL_EVERY_H=99 YGO_CAP_H=99 bash "$W/app/pause_test.sh" > "$W/logs/pause-test-r4.txt" 2>&1; then
      touch "$W/stage-r4-pause.done"; say "pause: $(tail -1 "$W/logs/pause-test-r4.txt")"; break
    fi
    say "pause: failed ($(tail -1 "$W/logs/pause-test-r4.txt"))"
  done
fi
say "pipeline-r4 done: $(free_gb) GB free, ledger $(ledger_h) GPU-h"
