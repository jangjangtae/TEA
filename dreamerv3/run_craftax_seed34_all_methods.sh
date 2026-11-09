#!/usr/bin/env bash
set -euo pipefail

# Complete the AAAI Craftax main comparison from 3 to 5 seeds by running
# seeds 3 and 4 for every method in the main figure:
#   No-adapt clean, Task-only, ScratchDreamer, Dreamer+RND, Dense surprise,
#   ExcessDelta, and Contextual excess.
#
# Existing seeds 0/1/2 are kept untouched. This queue writes only seed 3/4
# results under a new root and prunes replay buffers after each training run.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-$HOME/logdir/craftax_seed34_all_methods_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-3 4}"

BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-30000}"
ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-30000}"
SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-60000}"
MILESTONE_EVAL_STEPS="${MILESTONE_EVAL_STEPS:-10000}"
MILESTONE_SPARSE_EVAL_STEPS="${MILESTONE_SPARSE_EVAL_STEPS:-20000}"
TRAIN_ENVS="${TRAIN_ENVS:-16}"
TRAIN_RATIO="${TRAIN_RATIO:-128}"
REPLAY_SIZE="${REPLAY_SIZE:-100000}"
MIN_FREE_GB="${MIN_FREE_GB:-35}"

cd "$PROJECT_ROOT"
mkdir -p "$ROOT"
LAUNCHER_LOG="$ROOT/launcher.log"
STATUS_FILE="$ROOT/status.tsv"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

stamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

run_stage() {
  local name="$1"
  shift
  echo "===================================================="
  echo "[$(stamp)] $name START"
  echo "===================================================="
  echo -e "$(stamp)\t${name}\tSTART" | tee -a "$STATUS_FILE"
  "$@"
  echo -e "$(stamp)\t${name}\tDONE" | tee -a "$STATUS_FILE"
  echo "===================================================="
  echo "[$(stamp)] $name DONE"
  echo "===================================================="
}

echo "===================================================="
echo "[Craftax seed 3/4 all-method completion]"
echo "root       : $ROOT"
echo "train root : $TRAIN_ROOT"
echo "seeds      : $SEEDS"
echo "replay     : $REPLAY_SIZE, prune after train"
echo "min free GB: $MIN_FREE_GB"
echo "===================================================="

run_stage "01_noadapt_clean" env \
  ROOT="$ROOT/01_noadapt_clean" \
  TRAIN_ROOT="$TRAIN_ROOT" \
  WAIT_SERVICE= \
  RUN_IF_RND_FAILED=1 \
  SEEDS="$SEEDS" \
  BASE_EVAL_STEPS="$BASE_EVAL_STEPS" \
  SPARSE_EVAL_STEPS="$SPARSE_EVAL_STEPS" \
  MIN_FREE_GB="$MIN_FREE_GB" \
  "$PROJECT_ROOT/dreamerv3/run_craftax_cleaneval_after_rnd.sh"

run_stage "02_cleaninit_reward_design" env \
  ROOT="$ROOT/02_cleaninit_reward_design" \
  TRAIN_ROOT="$TRAIN_ROOT" \
  SEEDS="$SEEDS" \
  VARIANTS="taskonly dense_beta02 excess_delta_p95_beta02 contextual_excess_delta_beta02" \
  EVAL_SPLITS="clean seen holdout sparse" \
  TRAIN_STEPS=1000000 \
  TRAIN_MILESTONES=1000000 \
  TRAIN_ENVS="$TRAIN_ENVS" \
  TRAIN_RATIO="$TRAIN_RATIO" \
  REPLAY_SIZE="$REPLAY_SIZE" \
  BASE_EVAL_STEPS="$BASE_EVAL_STEPS" \
  ADAPT_EVAL_STEPS="$ADAPT_EVAL_STEPS" \
  SPARSE_EVAL_STEPS="$SPARSE_EVAL_STEPS" \
  MILESTONE_EVAL_STEPS="$MILESTONE_EVAL_STEPS" \
  MILESTONE_SPARSE_EVAL_STEPS="$MILESTONE_SPARSE_EVAL_STEPS" \
  MIN_FREE_GB="$MIN_FREE_GB" \
  PRUNE_REPLAY_AFTER_TRAIN=1 \
  STOP_ON_FAIL=1 \
  RUN_ANALYSIS=1 \
  "$PROJECT_ROOT/dreamerv3/run_craftax_multiseed_fault_queue.sh"

run_stage "03_cleaninit_rnd" env \
  ROOT="$ROOT/03_cleaninit_rnd" \
  TRAIN_ROOT="$TRAIN_ROOT" \
  SEEDS="$SEEDS" \
  VARIANTS="rnd_beta005" \
  EVAL_SPLITS="clean seen holdout sparse" \
  TRAIN_STEPS=1000000 \
  TRAIN_MILESTONES=1000000 \
  TRAIN_ENVS="$TRAIN_ENVS" \
  TRAIN_RATIO="$TRAIN_RATIO" \
  REPLAY_SIZE="$REPLAY_SIZE" \
  BASE_EVAL_STEPS="$BASE_EVAL_STEPS" \
  ADAPT_EVAL_STEPS="$ADAPT_EVAL_STEPS" \
  SPARSE_EVAL_STEPS="$SPARSE_EVAL_STEPS" \
  MILESTONE_EVAL_STEPS="$MILESTONE_EVAL_STEPS" \
  MILESTONE_SPARSE_EVAL_STEPS="$MILESTONE_SPARSE_EVAL_STEPS" \
  MIN_FREE_GB="$MIN_FREE_GB" \
  PRUNE_REPLAY_AFTER_TRAIN=1 \
  STOP_ON_FAIL=1 \
  RUN_ANALYSIS=1 \
  "$PROJECT_ROOT/dreamerv3/run_craftax_multiseed_fault_queue.sh"

run_stage "04_scratch_dreamer" env \
  ROOT="$ROOT/04_scratch_dreamer" \
  TRAIN_ROOT="$TRAIN_ROOT" \
  SEEDS="$SEEDS" \
  VARIANTS="bugonly_from_scratch" \
  EVAL_SPLITS="clean seen holdout sparse" \
  TRAIN_STEPS=2100000 \
  TRAIN_MILESTONES=2100000 \
  TRAIN_ENVS="$TRAIN_ENVS" \
  TRAIN_RATIO="$TRAIN_RATIO" \
  REPLAY_SIZE="$REPLAY_SIZE" \
  BASE_EVAL_STEPS="$BASE_EVAL_STEPS" \
  ADAPT_EVAL_STEPS="$ADAPT_EVAL_STEPS" \
  SPARSE_EVAL_STEPS="$SPARSE_EVAL_STEPS" \
  MILESTONE_EVAL_STEPS="$MILESTONE_EVAL_STEPS" \
  MILESTONE_SPARSE_EVAL_STEPS="$MILESTONE_SPARSE_EVAL_STEPS" \
  MIN_FREE_GB="$MIN_FREE_GB" \
  PRUNE_REPLAY_AFTER_TRAIN=1 \
  STOP_ON_FAIL=1 \
  RUN_ANALYSIS=1 \
  "$PROJECT_ROOT/dreamerv3/run_craftax_multiseed_fault_queue.sh"

echo "===================================================="
echo "[Craftax seed 3/4 all-method completion] DONE"
echo "root   : $ROOT"
echo "status : $STATUS_FILE"
echo "log    : $LAUNCHER_LOG"
echo "===================================================="
