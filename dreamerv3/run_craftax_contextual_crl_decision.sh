#!/usr/bin/env bash
set -euo pipefail

# Overnight decision probe. It can be launched while another queue is active:
# WAIT_PID is polled before this script claims the GPU.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-$HOME/logdir/craftax_contextual_crl_decision_$(date +%Y%m%d_%H%M%S)}"
WAIT_PID="${WAIT_PID:-}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

# XLA may resolve libdevice relative to the current working directory. Keep
# unattended systemd launches anchored to the repository where the verified
# libdevice.10.bc compatibility link lives.
cd "$PROJECT_ROOT"
mkdir -p "$ROOT"

exec > >(tee -a "$ROOT/launcher.log") 2>&1

echo "===================================================="
echo "[Contextual CRL decision probe]"
echo "root       : $ROOT"
echo "wait pid   : ${WAIT_PID:-none}"
echo "seeds      : ${SEEDS:-0 1 2}"
echo "variants   : ${VARIANTS:-taskonly contextual_excess_delta_beta02 contextual_crl_task85}"
echo "train steps: ${TRAIN_STEPS:-300000}"
echo "===================================================="

if [[ -n "$WAIT_PID" ]]; then
  echo "$(date '+%F %T') waiting for PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$WAIT_SECONDS"
  done
  echo "$(date '+%F %T') PID $WAIT_PID finished; starting decision probe"
fi

ROOT="$ROOT" TRAIN_ROOT="$TRAIN_ROOT" \
SEEDS="${SEEDS:-0 1 2}" \
VARIANTS="${VARIANTS:-taskonly contextual_excess_delta_beta02 contextual_crl_task85}" \
EVAL_SPLITS="clean seen holdout sparse" \
TRAIN_STEPS="${TRAIN_STEPS:-300000}" TRAIN_ENVS="16" TRAIN_RATIO="128" \
REPLAY_SIZE="100000" BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-20000}" \
ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-20000}" \
SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-40000}" \
CONSTRAINT_TASK_TARGET="${CONSTRAINT_TASK_TARGET:-1.45}" \
CONSTRAINT_WARMUP_EPISODES="${CONSTRAINT_WARMUP_EPISODES:-10}" \
MIN_FREE_GB="80" PRUNE_REPLAY_AFTER_TRAIN="1" STOP_ON_FAIL="1" \
  "$PROJECT_ROOT/dreamerv3/run_craftax_contextual_crl_queue.sh"

"$PROJECT_ROOT/dreamer_cuda/bin/python" \
  "$PROJECT_ROOT/dreamerv3/select_contextual_candidates.py" \
  --analysis "$ROOT/analysis" --task-retention 0.90 --top-k 2

echo "Contextual CRL decision probe DONE: $ROOT"
