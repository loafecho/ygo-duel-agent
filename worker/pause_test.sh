#!/bin/bash
# Pause proof on the GPU workstation: the self-play trainer runs under supervisor.py; a fake game (a copy of `sleep`
# named gamescope, run as `gamescope -- 120`, which the copied guard reports as a game) must make it
# checkpoint and free the GPU within 10 s; killing the fake game must bring it back from the same step
# within 60 s, then 10+ more steps. Run with no worker on the GPU (the baseline is taken first).
# The trainer is the self-play PPO trainer (worker.py train), playing real games on the game host; it runs in its
# own data dir (pause-data) so the real training run's checkpoints are never touched.
set -u
W=~/.local/share/ygo-worker
export YGO_DATA=$W/pause-data YGO_EVAL_GAMES=${YGO_EVAL_GAMES:-20}
L=$YGO_DATA/logs/train.jsonl
S=$W/logs/supervisor-train.jsonl
now() { date +%s.%N; }
since() { awk -v a="$(now)" -v b="$1" 'BEGIN { printf "%.1f", a - b }'; }
steps() { grep -c '"event": "step"' "$L"; }

rm -rf "$YGO_DATA"
mkdir -p "$YGO_DATA/logs"
: > "$L"
: > "$S"
base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "baseline_vram_mib=$base (before the worker starts)"

setsid "$W/app/supervisor.py" train > "$W/logs/supervisor-train.out" 2>&1 < /dev/null &
SUP=$!
until [ "$(steps)" -ge 20 ]; do sleep 1; done
WPID=$(grep '"event": "started"' "$S" | tail -1 | sed 's/.*"pid": \([0-9]*\).*/\1/')
echo "training: $(steps) steps, worker pid $WPID"
echo "gpu apps before game: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"

t0=$(now)
"$W/fake/gamescope" -- 120 &
FAKE=$!
echo "fake game started (pid $FAKE)"
freed=""
for _ in $(seq 1 60); do
  if grep -q '"event": "checkpoint"' "$L" && ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -qx "$WPID"; then
    freed=$(since "$t0"); break
  fi
  sleep 0.5
done
vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
ckpt=$(grep '"event": "checkpoint"' "$L" | tail -1 | sed 's/.*"msg": "\([^"]*\)".*/\1/')
echo "log: $ckpt"
echo "gpu freed after ${freed:-NOT FREED} s; worker pid $WPID on GPU: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -cx "$WPID")"
echo "nvidia-smi memory.used=${vram} MiB, baseline ${base} MiB, delta $((vram - base)) MiB"
(sleep 10; echo "diagnostic only: memory.used 10 s later=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB; apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')") &
echo "supervisor: $(grep '"event": "game_detected"\|"event": "stopped"' "$S" | tail -2 | tr '\n' ' ')"

sleep 5
N=${ckpt#checkpoint step=}
t1=$(now)
kill "$FAKE"
echo "fake game killed"
resumed=""
for _ in $(seq 1 120); do
  if grep -q "\"msg\": \"resumed step=$N\"" "$L"; then resumed=$(since "$t1"); break; fi
  sleep 0.5
done
echo "log: $(grep '"event": "resumed"' "$L" | tail -1 | sed 's/.*"msg": "\([^"]*\)".*/\1/') after ${resumed:-NOT RESUMED} s"
for _ in $(seq 1 120); do
  last=$(grep '"event": "step"' "$L" | tail -1 | sed 's/.*"step": \([0-9]*\).*/\1/')
  [ "${last:-0}" -ge $((N + 10)) ] && break
  sleep 0.5
done
echo "steps after resume: $((last - N)) (step $N -> $last)"

kill -TERM "$SUP"
wait "$SUP" 2>/dev/null
echo "supervisor stopped: $(tail -1 "$S")"
