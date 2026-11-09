#!/usr/bin/env bash
set -euo pipefail

# Wait until the GPU is idle, then launch the MiniGrid weekend queue. This is
# useful when a Craftax run is already occupying the GPU and MiniGrid should
# start automatically afterwards.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
ROOT="${ROOT:-$HOME/logdir/minigrid_weekend_fault_$(date +%Y%m%d_%H%M%S)}"

WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-1}"
GPU_IDLE_SECONDS="${GPU_IDLE_SECONDS:-600}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"
RUN_SMOKE_FIRST="${RUN_SMOKE_FIRST:-1}"

export ROOT
export CONFIGS="${CONFIGS:-minigrid}"
export SEEDS="${SEEDS:-0 1 2}"
export VARIANTS="${VARIANTS:-taskonly dense_beta02 excess_delta_p95_beta02 contextual_excess_delta_beta02}"
export EVAL_SPLITS="${EVAL_SPLITS:-clean seen holdout sparse}"

export CLEAN_TRAIN_STEPS="${CLEAN_TRAIN_STEPS:-500000}"
export TRAIN_STEPS="${TRAIN_STEPS:-500000}"
export TRAIN_MILESTONES="${TRAIN_MILESTONES:-100000 300000 500000}"
export TRAIN_ENVS="${TRAIN_ENVS:-16}"
export TRAIN_RATIO="${TRAIN_RATIO:-128}"
export REPLAY_SIZE="${REPLAY_SIZE:-50000}"

export BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-20000}"
export ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-20000}"
export SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-40000}"
export MILESTONE_EVAL_STEPS="${MILESTONE_EVAL_STEPS:-10000}"
export MILESTONE_SPARSE_EVAL_STEPS="${MILESTONE_SPARSE_EVAL_STEPS:-20000}"

export MIN_FREE_GB="${MIN_FREE_GB:-25}"
export PRUNE_REPLAY_AFTER_TRAIN="${PRUNE_REPLAY_AFTER_TRAIN:-1}"
export STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
export RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

mkdir -p "$ROOT"
LAUNCHER_LOG="$ROOT/wait_launcher.log"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

stamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

gpu_busy_count() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
    awk 'NF {count += 1} END {print count + 0}'
}

echo "===================================================="
echo "[MiniGrid weekend launcher]"
echo "root              : $ROOT"
echo "wait for gpu idle : $WAIT_FOR_GPU_IDLE"
echo "idle seconds      : $GPU_IDLE_SECONDS"
echo "poll seconds      : $GPU_POLL_SECONDS"
echo "smoke first       : $RUN_SMOKE_FIRST"
echo "configs           : $CONFIGS"
echo "seeds             : $SEEDS"
echo "variants          : $VARIANTS"
echo "clean/adapt steps : $CLEAN_TRAIN_STEPS / $TRAIN_STEPS"
echo "===================================================="

if [[ "$WAIT_FOR_GPU_IDLE" == "1" ]]; then
  idle=0
  while (( idle < GPU_IDLE_SECONDS )); do
    busy="$(gpu_busy_count)"
    if (( busy == 0 )); then
      idle=$((idle + GPU_POLL_SECONDS))
      echo "$(stamp) GPU idle for ${idle}/${GPU_IDLE_SECONDS}s"
    else
      idle=0
      echo "$(stamp) GPU busy processes=${busy}; waiting ${GPU_POLL_SECONDS}s"
    fi
    if (( idle < GPU_IDLE_SECONDS )); then
      sleep "$GPU_POLL_SECONDS"
    fi
  done
fi

if [[ "$RUN_SMOKE_FIRST" == "1" ]]; then
  echo "$(stamp) running MiniGrid GPU queue smoke"
  ROOT="$ROOT/smoke" \
  SEEDS="0" \
  VARIANTS="taskonly" \
  EVAL_SPLITS="clean seen" \
  CLEAN_TRAIN_STEPS="64" \
  TRAIN_STEPS="64" \
  TRAIN_MILESTONES="64" \
  TRAIN_ENVS="1" \
  TRAIN_RATIO="1" \
  REPLAY_SIZE="1000" \
  SAVE_EVERY="32" \
  BASE_EVAL_STEPS="20" \
  ADAPT_EVAL_STEPS="20" \
  SPARSE_EVAL_STEPS="20" \
  MILESTONE_EVAL_STEPS="20" \
  MILESTONE_SPARSE_EVAL_STEPS="20" \
  BATCH_SIZE="2" \
  BATCH_LENGTH="8" \
  REPORT_LENGTH="8" \
  EVAL_ENVS="1" \
  MIN_FREE_GB="$MIN_FREE_GB" \
  RUN_ANALYSIS="1" \
  "$PROJECT_ROOT/dreamerv3/run_minigrid_multiseed_fault_queue.sh"
  echo "$(stamp) MiniGrid GPU queue smoke completed"
fi

echo "$(stamp) launching MiniGrid queue"
exec "$PROJECT_ROOT/dreamerv3/run_minigrid_multiseed_fault_queue.sh"
