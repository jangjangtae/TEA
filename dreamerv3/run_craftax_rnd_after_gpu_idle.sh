#!/usr/bin/env bash
set -euo pipefail

# Wait for the current GPU job to finish, then run the DreamerV3+RND
# adaptation baseline. This tests whether a generic novelty reward can match
# the clean-prior excess-delta method.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-$HOME/logdir/craftax_rnd_baseline_$(date +%Y%m%d_%H%M%S)}"

GPU_IDLE_SECONDS="${GPU_IDLE_SECONDS:-600}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
RUN_SMOKE_FIRST="${RUN_SMOKE_FIRST:-1}"

cd "$PROJECT_ROOT"
mkdir -p "$ROOT"

echo "===================================================="
echo "[Craftax RND baseline after GPU idle]"
echo "root          : $ROOT"
echo "train root    : $TRAIN_ROOT"
echo "idle seconds  : $GPU_IDLE_SECONDS"
echo "check interval: $CHECK_INTERVAL"
echo "run smoke     : $RUN_SMOKE_FIRST"
echo "variants      : ${VARIANTS:-rnd_beta005}"
echo "===================================================="

idle_start=""
while true; do
  procs="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  now="$(date +%s)"
  if [[ -z "${procs// }" ]]; then
    if [[ -z "$idle_start" ]]; then
      idle_start="$now"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU appears idle; starting idle timer."
    fi
    idle_for=$((now - idle_start))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU idle for ${idle_for}s / ${GPU_IDLE_SECONDS}s"
    if (( idle_for >= GPU_IDLE_SECONDS )); then
      break
    fi
  else
    idle_start=""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU busy:"
    echo "$procs"
  fi
  sleep "$CHECK_INTERVAL"
done

if [[ "$RUN_SMOKE_FIRST" == "1" ]]; then
  echo "===================================================="
  echo "[RND smoke]"
  echo "===================================================="
  ROOT="$ROOT/smoke" \
  TRAIN_ROOT="$TRAIN_ROOT" \
  SEEDS="${SMOKE_SEEDS:-0}" \
  VARIANTS="${SMOKE_VARIANTS:-rnd_beta005}" \
  EVAL_SPLITS="${SMOKE_EVAL_SPLITS:-clean}" \
  TRAIN_STEPS="${SMOKE_TRAIN_STEPS:-64}" \
  TRAIN_MILESTONES="" \
  BASE_EVAL_STEPS="${SMOKE_BASE_EVAL_STEPS:-32}" \
  ADAPT_EVAL_STEPS="${SMOKE_ADAPT_EVAL_STEPS:-32}" \
  SPARSE_EVAL_STEPS="${SMOKE_SPARSE_EVAL_STEPS:-32}" \
  TRAIN_ENVS="${SMOKE_TRAIN_ENVS:-1}" \
  TRAIN_RATIO="${SMOKE_TRAIN_RATIO:-1}" \
  REPLAY_SIZE="${SMOKE_REPLAY_SIZE:-256}" \
  BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}" \
  BATCH_LENGTH="${SMOKE_BATCH_LENGTH:-8}" \
  REPORT_LENGTH="${SMOKE_REPORT_LENGTH:-8}" \
  EVAL_ENVS="${SMOKE_EVAL_ENVS:-1}" \
  MIN_FREE_GB="${MIN_FREE_GB:-35}" \
  RUN_ANALYSIS=0 \
  RUN_BASE_EVALS=1 \
  PRUNE_REPLAY_AFTER_TRAIN=1 \
  "$PROJECT_ROOT/dreamerv3/run_craftax_multiseed_fault_queue.sh"
fi

echo "===================================================="
echo "[RND full run]"
echo "===================================================="
ROOT="$ROOT/full" \
TRAIN_ROOT="$TRAIN_ROOT" \
SEEDS="${SEEDS:-0 1 2}" \
VARIANTS="${VARIANTS:-rnd_beta005}" \
EVAL_SPLITS="${EVAL_SPLITS:-clean seen holdout sparse}" \
TRAIN_STEPS="${TRAIN_STEPS:-1000000}" \
TRAIN_MILESTONES="${TRAIN_MILESTONES:-1000000}" \
TRAIN_ENVS="${TRAIN_ENVS:-16}" \
TRAIN_RATIO="${TRAIN_RATIO:-128}" \
REPLAY_SIZE="${REPLAY_SIZE:-100000}" \
BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-30000}" \
ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-30000}" \
SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-60000}" \
MILESTONE_EVAL_STEPS="${MILESTONE_EVAL_STEPS:-10000}" \
MILESTONE_SPARSE_EVAL_STEPS="${MILESTONE_SPARSE_EVAL_STEPS:-20000}" \
MIN_FREE_GB="${MIN_FREE_GB:-35}" \
STOP_ON_FAIL="${STOP_ON_FAIL:-1}" \
RUN_ANALYSIS="${RUN_ANALYSIS:-1}" \
PRUNE_REPLAY_AFTER_TRAIN="${PRUNE_REPLAY_AFTER_TRAIN:-1}" \
"$PROJECT_ROOT/dreamerv3/run_craftax_multiseed_fault_queue.sh"

echo "===================================================="
echo "[Craftax RND baseline after GPU idle] DONE"
echo "root: $ROOT"
echo "===================================================="
