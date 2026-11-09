#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
WAIT_PID="${WAIT_PID:?Set WAIT_PID to the process id to wait for.}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUEUE_LOG="${QUEUE_LOG:-$HOME/logdir/craftax_ratio512_queue_launcher.log}"

mkdir -p "$(dirname "$QUEUE_LOG")"
exec >> "$QUEUE_LOG" 2>&1

echo "[queue] $(date '+%F %T') waiting for PID $WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

echo "[queue] $(date '+%F %T') previous run finished; starting ratio512 run"
cd "$PROJECT_ROOT"

ROOT="${ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_$(date +%Y%m%d_%H%M%S)}" \
STEPS="${STEPS:-1100000}" \
ENVS="${ENVS:-16}" \
TRAIN_RATIO="${TRAIN_RATIO:-512}" \
REPLAY_SIZE="${REPLAY_SIZE:-200000}" \
SAVE_EVERY="${SAVE_EVERY:-200000}" \
"$PROJECT_ROOT/dreamerv3/run_craftax_speed_probe.sh"

echo "[queue] $(date '+%F %T') ratio512 run finished"
