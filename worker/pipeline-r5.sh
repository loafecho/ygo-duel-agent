#!/bin/bash
# Round 5 on the GPU workstation: pre-registered baseline (+ a 5-deck WindBot pool), research (notes, no GPU),
# a deck-ceiling test (WindBot vs WindBot on the game host, no GPU), gated tries of a 1.5B base, DAgger round 2, a bigger
# critic and new rewards, exploring practice vs the deck pool, and a pre-registered final.
# Same pattern as pipeline-r4.sh: every GPU step runs under supervisor.py (the game guard); a finished stage leaves
# stage-r5-<name>.done; control-host and game-host steps leave marker files this script waits for.
# Every eval is greedy, half the games as first player, draws count as losses, Wilson 95% intervals.
# GPU-h ledger: data-r5/ledger.jsonl; 48 GPU-h cap, 3 reserved for the final. Log: logs/pipeline-r5.log
set -u
W=~/.local/share/ygo-worker
D=$W/data-r5
export HF_HOME=$W/hf YGO_CARD_TEXT=${YGO_CARD_TEXT:-oneliner} YGO_HISTORY=1 YGO_ONELINERS=${YGO_ONELINERS:-oneliners-r5.json}
LOG=$W/logs/pipeline-r5.log
R4=$W/data-r4/sl-dagger/final.pt
CAP_H=48 RESERVE_H=3
BROKER=${YGO_BROKER:-http://198.51.100.10:8765}
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
setv() { echo "$2" > "$D/$1"; }
getv() { cat "$D/$1"; }
max() { py "print(max($1, $2))"; }
gt() { py "import sys; sys.exit(0 if $1 > $2 else 1)"; }
wait_file() {  # a control-host or game-host step (runner deploy, labels, research) leaves this file
  [ -e "$1" ] || say "waiting for $1 ($2)"
  until [ -e "$1" ]; do sleep 30; done
}

match() {  # match <name> <A.pt> <B: windbot | pool | .pt> <games> <seed>: one supervised match in $D/m-<name>
  local dir=$D/m-$1
  if [ ! -e "$dir/done-match" ]; then
    wait_idle
    say "match $1: start ($(basename "$2") vs $(basename "$3"), $4 games)"
    mkdir -p "$dir/logs"
    YGO_DATA=$dir YGO_MATCH_NAME=r5-$1 YGO_MATCH_A=$2 YGO_MATCH_B=$3 YGO_MATCH_GAMES=$4 YGO_MATCH_SEED=$5 \
      YGO_MATCH_VALUE_LOG=${YGO_MATCH_VALUE_LOG-$D/value-model.jsonl} \
      /usr/bin/python3 -I "$W/app/supervisor.py" match >> "$W/logs/supervisor-match-r5.out" 2>&1 < /dev/null
    [ -e "$dir/done-match" ] || { say "match $1: supervisor exited without done-match"; return 1; }
    ledger_add "match-$1" "$(field "$dir/result.json" gpu_s)"
  fi
}
ci() { field "$1" "f'{win_rate:.3f} ci95=[{ci95[0]:.3f},{ci95[1]:.3f}] games={games}'"; }
sup() {  # sup <mode> <dir>: one supervised run of a worker mode in its own data dir (env passed through)
  wait_idle
  mkdir -p "$2/logs"
  YGO_DATA=$2 /usr/bin/python3 -I "$W/app/supervisor.py" "$1" >> "$W/logs/supervisor-$1-r5.out" 2>&1 < /dev/null
  [ -e "$2/done-$1" ]
}

mkdir -p "$D/logs"
touch "$D/ledger.jsonl"
say "pipeline-r5 up: $(free_gb) GB free, ledger $(ledger_h) GPU-h, card_text=$YGO_CARD_TEXT oneliners=$YGO_ONELINERS"

# 0. Cleanup check: the game server was restarted on the game host (round 4 left running=96 pending=96 behind).
if [ ! -e "$W/stage-r5-cleanup.done" ]; then
  ST=$(curl -s --max-time 10 "$BROKER/status")
  R=$(echo "$ST" | py 'import json,sys; s=json.load(sys.stdin); print("running={running} pending={pending} queued={queued}".format(**s))')
  say "ygo-server /status $R"
  case "$R" in running=0\ *) touch "$W/stage-r5-cleanup.done" ;; *) say "cleanup: server not idle"; exit 1 ;; esac
fi

# 1. Baseline, pre-registered: round 4's best on the round-4 setup (500 games) and vs the 5-deck pool (300 games).
if [ ! -e "$W/stage-r5-baseline.done" ]; then
  match baseline-wb "$R4" windbot 500 51001 || exit 1
  wait_file "$W/stage-r5-runner.done" "the round-5 runner (per-seat decks) deployed on the game host, pool decks fixed"
  match baseline-pool "$R4" pool 300 51002 || exit 1
  say "baseline vs_windbot=$(ci "$D/m-baseline-wb/result.json") vs_pool=$(field "$D/m-baseline-pool/result.json" "f'{win_rate:.3f} games={games}'")"
  say "baseline vs_pool by deck: $(field "$D/m-baseline-pool/result.json" by_deck)"
  setv current "$R4"; setv current_wr "$(field "$D/m-baseline-wb/result.json" win_rate)"
  setv best_wr "$(getv current_wr)"; setv model 0.5b
  touch "$W/stage-r5-baseline.done"
fi
say "current $(getv current) model=$(getv model) current_wr=$(getv current_wr) best_wr=$(getv best_wr)"

# 2. Research (control host, web, no GPU): a written research summary; the control host leaves this marker.
wait_file "$W/stage-r5-research.done" "the Round 5 research summary"
say "$(grep -h 'research sources=' "$W/stage-r5-research.done")"

# 3. Deck ceiling (game host, CPU): WindBot pilots our deck and each candidate vs the pool. A switch (a candidate beats
#    ours by >= 0.05) needs one-liners, the leak check, new WindBot labels, re-imitation + DAgger and a 500-game eval.
wait_file "$W/stage-r5-ceiling.done" "ceiling lines from the game host (ygo-runner ceiling, 300 games per deck)"
grep -h '^ceiling \|^deck=\|^oneliners \|^leakcheck ' "$W/stage-r5-ceiling.done" | while read -r l; do say "$l"; done
DECKSEL=$(grep -h '^deck=' "$W/stage-r5-ceiling.done" | tail -1 | sed 's/^deck=//; s/ .*//')
if [ ! -e "$W/stage-r5-deck.done" ]; then
  case "$DECKSEL" in
    kept) setv deck "" ;;
    switched:*)
      NEWDECK=${DECKSEL#switched:}
      setv deck "$NEWDECK"
      export YGO_DECK=$NEWDECK
      wait_file "$W/labels-r5deck.jsonl" "WindBot $NEWDECK self-play labels from the game host"
      if [ ! -e "$D/sl-deck/done-sl" ]; then
        YGO_WARM=$(getv current) YGO_LABELS_FILE=$W/labels-r5deck.jsonl YGO_SL_LR=1e-4 sup sl "$D/sl-deck" \
          || { say "deck sl: supervisor exited without done-sl"; exit 1; }
        ledger_add sl-deck "$(field "$D/sl-deck/result.json" gpu_s)"
        say "deck re-imitation: heldout_acc=$(field "$D/sl-deck/result.json" heldout_acc)"
      fi
      YGO_MATCH_SHADOW=1 match deck-dagger-collect "$D/sl-deck/final.pt" windbot 300 53001 || exit 1
      R=$D/m-deck-dagger-collect/result.json
      say "deck dagger games=$(field "$R" games) coverage=$(field "$R" coverage) disagree=$(field "$R" disagree)"
      wait_file "$W/labels-r5deck-dagger.jsonl" "deck DAgger labels aggregated on the control host"
      if [ ! -e "$D/sl-deck-dagger/done-sl" ]; then
        YGO_WARM=$D/sl-deck/final.pt YGO_LABELS_FILE=$W/labels-r5deck-dagger.jsonl YGO_SL_LR=3e-5 sup sl "$D/sl-deck-dagger" \
          || { say "deck dagger sl: supervisor exited without done-sl"; exit 1; }
        ledger_add sl-deck-dagger "$(field "$D/sl-deck-dagger/result.json" gpu_s)"
      fi
      match deck-eval "$D/sl-deck-dagger/final.pt" windbot 500 53002 || exit 1
      say "deck_eval deck=$NEWDECK vs_windbot=$(ci "$D/m-deck-eval/result.json") (becomes best-so-far)"
      setv current "$D/sl-deck-dagger/final.pt"; setv current_wr "$(field "$D/m-deck-eval/result.json" win_rate)"
      setv best_wr "$(getv current_wr)"; setv labels "$W/labels-r5deck-dagger.jsonl"
      ;;
    *) say "ceiling: no deck= line"; exit 1 ;;
  esac
  [ -e "$D/labels" ] || setv labels "$W/labels-dagger.jsonl"
  touch "$W/stage-r5-deck.done"
fi
YGO_DECK=$(getv deck); export YGO_DECK
say "deck=${YGO_DECK:-CyberDragon} current $(getv current) current_wr=$(getv current_wr) best_wr=$(getv best_wr) labels=$(getv labels)"

# 4. Bigger base model: Qwen2.5-1.5B-Instruct + a fresh LoRA, imitating the same labels (<= 8 GPU-h), 500-game eval.
M15=Qwen/Qwen2.5-1.5B-Instruct
if [ ! -e "$W/stage-r5-model.done" ]; then
  FREE=$(free_gb)
  if [ ! -d "$W/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct" ] && [ "$FREE" -lt 14 ]; then
    say "model_trial skipped reason=disk (${FREE} GB free; the ~3.1 GB download would leave < 10 GB)"
    setv model 0.5b
  else
    if [ ! -e "$D/m15-downloaded" ]; then
      say "model_trial: downloading $M15 ($FREE GB free)"
      "$W/venv/bin/python" -I -c "from huggingface_hub import snapshot_download; snapshot_download('$M15', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json'])" \
        >> "$W/logs/m15-download.out" 2>&1 || { say "model_trial: download failed"; exit 1; }
      touch "$D/m15-downloaded"
      say "model_trial: downloaded, $(free_gb) GB free"
    fi
    if [ "$(free_gb)" -lt 10 ]; then say "model_trial: below the 10 GB floor"; exit 1; fi
    if [ ! -e "$D/sl-15b/done-sl" ]; then
      YGO_MODEL=$M15 YGO_WARM='' YGO_LABELS_FILE=$(getv labels) YGO_SL_LR=1e-4 YGO_SL_CAP_H=6.5 YGO_MICRO=2 sup sl "$D/sl-15b" \
        || { say "1.5b sl: supervisor exited without done-sl"; exit 1; }
      ledger_add sl-15b "$(field "$D/sl-15b/result.json" gpu_s)"
      say "1.5b imitation: heldout_acc=$(field "$D/sl-15b/result.json" heldout_acc) steps=$(field "$D/sl-15b/result.json" steps)"
    fi
    YGO_MODEL=$M15 match m15-eval "$D/sl-15b/final.pt" windbot 500 54001 || exit 1
    MX=$(field "$D/m-m15-eval/result.json" win_rate)
    say "model_trial base=Qwen2.5-1.5B eval=$MX games=$(field "$D/m-m15-eval/result.json" games) ci95=$(field "$D/m-m15-eval/result.json" ci95) (best so far $(getv best_wr))"
    if gt "$MX" "$(getv best_wr)"; then
      setv model 1.5b; setv current "$D/sl-15b/final.pt"; setv current_wr "$MX"; setv best_wr "$MX"
    else
      setv model 0.5b
    fi
  fi
  say "model=$(getv model)"
  touch "$W/stage-r5-model.done"
fi
if [ "$(getv model)" = 1.5b ]; then export YGO_MODEL=$M15 YGO_MICRO=2; fi
say "model=$(getv model) current $(getv current) best_wr=$(getv best_wr)"
# Round 5 budget fix (2026-10-26): the final's reserve is sized from the kept model's own 500-game eval cost
# (the 3 GPU-h reserve assumed the 0.5B; the 1.5B plays ~2.6x slower). EV = GPU-h per 500 greedy games.
EVF=$D/m-deck-eval/result.json; [ "$(getv model)" = 1.5b ] && EVF=$D/m-m15-eval/result.json
EV=$(py "print(round($(field "$EVF" gpu_s) / 3600, 3))")
FINAL_H=$(py "print(round($EV * 1100 / 500, 2))")
RESERVE_H=$(py "print(max(3.0, $FINAL_H + 0.5))")
fits() { py "import sys; sys.exit(0 if $1 <= $(left_h) else 1)"; }  # fits <GPU-h>: room before the final's reserve
say "budget: ledger $(ledger_h), EV=$EV GPU-h per 500 games, final ~$FINAL_H + pause 0.5 -> reserve $RESERVE_H, $(left_h) GPU-h left for stages 5-8"

# 5. DAgger round 2 on the kept model: 300 shadow games, aggregate, re-imitate, 500-game eval; kept iff it beats best.
NEED=$(py "print(round($EV * 800 / 500 + ${YGO_SL_CAP_H:-3}, 2))")
if [ ! -e "$W/stage-r5-dagger2.done" ] && [ ! -e "$D/m-dagger2-collect/done-match" ] && ! fits "$NEED"; then
  if [ -e "$D/m-dagger2-collect/gpu_s.json" ]; then ledger_add dagger2-collect-partial "$(field "$D/m-dagger2-collect/gpu_s.json" gpu_s)"; fi
  say "dagger2 skipped reason=budget (collect 300 + imitation + 500-game eval ~$NEED GPU-h > $(left_h) left before the final's reserve)"
  say "dagger2=skipped (not run; current model unchanged)"
  touch "$W/stage-r5-dagger2.done"
fi
if [ ! -e "$W/stage-r5-dagger2.done" ]; then
  YGO_MATCH_SHADOW=1 match dagger2-collect "$(getv current)" windbot 300 55001 || exit 1
  R=$D/m-dagger2-collect/result.json
  wait_file "$W/labels-dagger2.jsonl" "round-2 DAgger labels aggregated on the control host"
  if [ ! -e "$D/sl-dagger2/done-sl" ]; then
    YGO_WARM=$(getv current) YGO_LABELS_FILE=$W/labels-dagger2.jsonl YGO_SL_LR=3e-5 sup sl "$D/sl-dagger2" \
      || { say "dagger2 sl: supervisor exited without done-sl"; exit 1; }
    ledger_add sl-dagger2 "$(field "$D/sl-dagger2/result.json" gpu_s)"
  fi
  match dagger2-eval "$D/sl-dagger2/final.pt" windbot 500 55002 || exit 1
  DX=$(field "$D/m-dagger2-eval/result.json" win_rate)
  say "dagger2 games=$(field "$R" games) coverage=$(field "$R" coverage) disagree=$(field "$R" disagree) eval=$DX games=$(field "$D/m-dagger2-eval/result.json" games) ci95=$(field "$D/m-dagger2-eval/result.json" ci95) (best so far $(getv best_wr))"
  if gt "$DX" "$(getv best_wr)"; then
    say "dagger2=kept"; setv current "$D/sl-dagger2/final.pt"; setv current_wr "$DX"; setv best_wr "$DX"
  else
    say "dagger2=reverted ($DX <= best $(getv best_wr))"
  fi
  touch "$W/stage-r5-dagger2.done"
fi

# 6. Bigger critic: a value head on its own trainable LoRA trunk, >= 4000 games of outcomes, round 4's gate.
if [ ! -e "$W/stage-r5-values.done" ]; then
  wait_file "$W/value-r5.jsonl" "WindBot self-play critic data (value-r5.jsonl) from the game host"
  VF=$W/value-r4.jsonl,$W/data-r4/value-model.jsonl,$D/value-model.jsonl,$W/value-r5.jsonl
  PG=16; [ "$(getv model)" = 1.5b ] && PG=3  # 1.5B: 3 decisions a game keeps >= 4000 games inside the budget
  if [ ! -e "$D/values/done-values2" ]; then
    YGO_VALUE_PER_GAME=$PG YGO_WARM=$(getv current) YGO_VALUE_FILES=$VF sup values2 "$D/values" || { say "values2: supervisor exited without done-values2"; exit 1; }
    ledger_add values2 "$(field "$D/values/result.json" gpu_s)"
  fi
  say "value auc=$(field "$D/values/result.json" auc) brier=$(field "$D/values/result.json" brier) brier_lp=$(field "$D/values/result.json" brier_lp) games=$(field "$D/values/result.json" games) (held-out $(field "$D/values/result.json" held_games) games)"
  say "critic=$(field "$D/values/result.json" critic) (on iff auc >= 0.75 and brier <= 0.9 x brier_lp)"
  touch "$W/stage-r5-values.done"
fi
CRITIC=$(field "$D/values/result.json" critic)
VH="" VT=""
[ "$CRITIC" = on ] && VH=$D/values/vhead.pt VT=$D/values/vtrunk.pt
export YGO_R5=1 YGO_ENT_FLOOR=${YGO_ENT_FLOOR:-0.25}

# 7. Reward trial: two matched practice slices (YGO_TRIAL_H GPU-h each, default 3) from the kept model (card-advantage potential + fast-win),
#    with vs without the response bonus; a 300-game eval each. response_bonus=on iff eval >= control.
NEED=$(py "print(round(2 * (${YGO_TRIAL_H:-3} + $EV * 300 / 500), 2))")
if [ ! -e "$W/stage-r5-reward.done" ] && [ ! -e "$D/trial-resp" ] && ! fits "$NEED"; then
  say "reward_trial skipped reason=budget (2 slices of ${YGO_TRIAL_H:-3} GPU-h + 2 x 300-game evals ~$NEED GPU-h > $(left_h) left)"
  setv response_bonus off; say "response_bonus=off (trial skipped)"
  touch "$W/stage-r5-reward.done"
fi
if [ ! -e "$W/stage-r5-reward.done" ]; then
  for arm in resp ctrl; do
    if [ ! -e "$D/trial-$arm/done-practice" ]; then
      RB=0; [ $arm = resp ] && RB=1
      YGO_WARM=$(getv current) YGO_VHEAD=$VH YGO_VTRUNK=$VT YGO_CAP_H=${YGO_TRIAL_H:-3} YGO_EVAL_EVERY_H=99 YGO_START_WR=$(getv current_wr) \
        YGO_RESPONSE_BONUS=$RB YGO_PRACTICE_SEED=57001 sup practice "$D/trial-$arm" || { say "trial $arm: supervisor exited without done-practice"; exit 1; }
      ledger_add "trial-$arm" "$(field "$D/trial-$arm/result.json" gpu_s)"
    fi
  done
  X=$(field "$D/trial-resp/result.json" "evals[-1]['vs_windbot']")
  Y=$(field "$D/trial-ctrl/result.json" "evals[-1]['vs_windbot']")
  say "reward_trial eval=$X control=$Y games=$(field "$D/trial-resp/result.json" "evals[-1]['games']") (entropy $(field "$D/trial-resp/result.json" "evals[-1]['entropy']") / $(field "$D/trial-ctrl/result.json" "evals[-1]['entropy']"))"
  if py "import sys; sys.exit(0 if $X >= $Y else 1)"; then setv response_bonus on; ARM=resp; else setv response_bonus off; ARM=ctrl; fi
  say "response_bonus=$(getv response_bonus)"
  ARMWR=$(field "$D/trial-$ARM/result.json" "evals[-1]['vs_windbot']")
  if gt "$ARMWR" "$(getv best_wr)"; then  # the 300-game slice eval beat best-so-far: practice continues from it
    setv current "$D/trial-$ARM/ckpt/pool/$(field "$D/trial-$ARM/result.json" "evals[-1]['ckpt']").pt"; setv current_wr "$ARMWR"
    say "practice starts from the $ARM slice ($ARMWR > $(getv best_wr))"
  fi
  touch "$W/stage-r5-reward.done"
fi
RB=0; [ "$(getv response_bonus)" = on ] && RB=1

# 8. Exploring practice: the rest of the budget minus the final's 3 GPU-h. Evals every 3 GPU-h print the entropy.
NEED=$(py "print(round(1 + $EV * 300 / 500, 2))")
if [ ! -e "$W/stage-r5-practice.done" ] && [ ! -e "$D/practice" ] && ! fits "$NEED"; then
  say "practice skipped reason=budget (>= 1 GPU-h of practice + a 300-game eval ~$NEED GPU-h > $(left_h) left)"
  touch "$W/stage-r5-practice.done"
fi
if [ ! -e "$W/stage-r5-practice.done" ]; then
  [ -e "$D/practice-cap" ] || setv practice-cap "$(py "print(round(max(0.0, $(left_h) - $EV * 300 / 500 - 0.3), 2))")"
  say "practice: cap $(getv practice-cap) GPU-h, critic=$CRITIC, response_bonus=$(getv response_bonus), ent_floor=$YGO_ENT_FLOOR, start $(getv current) at $(getv current_wr)"
  YGO_WARM=$(getv current) YGO_VHEAD=$VH YGO_VTRUNK=$VT YGO_CAP_H=$(getv practice-cap) YGO_START_WR=$(getv current_wr) \
    YGO_RESPONSE_BONUS=$RB YGO_PRACTICE_SEED=58001 sup practice "$D/practice" || { say "practice: supervisor exited without done-practice"; exit 1; }
  ledger_add practice "$(field "$D/practice/result.json" gpu_s)"
  py 'import json,sys; [print(f"eval step={e[\"step\"]} gpu_h={e[\"gpu_h\"]} vs_windbot={e[\"vs_windbot\"]:.3f} games={e[\"games\"]} entropy={e.get(\"entropy\")}") for e in json.load(open(sys.argv[1]))["evals"]]' \
    "$D/practice/result.json" | while read -r l; do say "$l"; done
  say "train_stop reason=$(cat "$D/practice/done-practice")"
  say "practice best: $(field "$D/practice/result.json" best)"
  touch "$W/stage-r5-practice.done"
fi

# 9. Final, pre-registered: 500 games on the round-4 setup, 300 paired games vs the round-4 model (each side its
#    own deck), 300 vs the pool. success = X >= 0.60, lo > 0.50, vs_r4 lo > 0.50; partial = X >= 0.55, lo > 0.50.
if [ ! -e "$W/stage-r5-final.done" ]; then
  if [ -e "$D/practice/result.json" ]; then
    FIN=$(field "$D/practice/result.json" "best['ckpt']"); STEP=$(field "$D/practice/result.json" "best['step']")
  else  # practice skipped: the final plays the kept model; its step is its imitation step count
    FIN=$(getv current); STEP=$(field "$(dirname "$FIN")/result.json" "steps")
  fi
  say "final: $FIN (step $STEP), $(left_h) GPU-h left before the reserve, ledger $(ledger_h)"
  match final-wb "$FIN" windbot 500 59001 || exit 1
  YGO_MATCH_PAIRED=1 YGO_MATCH_B_MODEL=Qwen/Qwen2.5-0.5B-Instruct match final-r4 "$FIN" "$R4" 300 59002 || exit 1
  match final-pool "$FIN" pool 300 59003 || exit 1
  A=$D/m-final-wb/result.json B=$D/m-final-r4/result.json C=$D/m-final-pool/result.json
  ERR=$(py "print($(field "$A" errors) + $(field "$B" errors) + $(field "$C" errors))")
  say "final step=$STEP vs_windbot=$(field "$A" "f'{win_rate:.3f} ci95=[{ci95[0]:.3f},{ci95[1]:.3f}] games={games}'") vs_r4=$(ci "$B") vs_pool=$(field "$C" "f'{win_rate:.3f} games={games}'") errors=$ERR"
  say "final vs_pool by deck: $(field "$C" by_deck); mean turns vs WindBot $(field "$A" mean_turns)"
  V=$(py "x,lo,r4=$(field "$A" win_rate),$(field "$A" 'ci95[0]'),$(field "$B" 'ci95[0]'); print('success' if x >= 0.60 and lo > 0.50 and r4 > 0.50 and $ERR == 0 else 'partial' if x >= 0.55 and lo > 0.50 else 'miss')")
  say "verdict=$V"; setv verdict "$V"
  touch "$W/stage-r5-final.done"
fi

# 10. Pause test against the round-5 trainer (practice, recipe 5), criteria unchanged. Its GPU time goes on the ledger.
if [ ! -e "$W/stage-r5-pause.done" ]; then
  for attempt in 1 2 3; do
    wait_idle
    say "pause: attempt $attempt (round-5 trainer: practice, recipe 5)"
    t0=$(date +%s)
    if YGO_PAUSE_MODE=practice YGO_PAUSE_DATA=$W/pause-data-r5 YGO_WARM=$(getv current) YGO_START_WR=$(getv current_wr) \
       YGO_RESPONSE_BONUS=$RB YGO_COLLECT=256 YGO_MINI=64 YGO_EVAL_EVERY_H=99 YGO_CAP_H=99 bash "$W/app/pause_test.sh" > "$W/logs/pause-test-r5.txt" 2>&1; then
      ledger_add "pause-$attempt" "$(( $(date +%s) - t0 ))"
      touch "$W/stage-r5-pause.done"; say "pause: $(tail -1 "$W/logs/pause-test-r5.txt")"; break
    fi
    ledger_add "pause-$attempt" "$(( $(date +%s) - t0 ))"
    say "pause: failed ($(tail -1 "$W/logs/pause-test-r5.txt"))"
  done
fi
say "pipeline-r5 done: $(free_gb) GB free, ledger $(ledger_h) GPU-h"
