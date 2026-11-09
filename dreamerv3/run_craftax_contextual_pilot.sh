#!/usr/bin/env bash
set -euo pipefail

# One-day feasibility gate: one seed, all reward ablations, moderate training.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-$HOME/logdir/craftax_contextual_pilot_$(date +%Y%m%d_%H%M%S)}"

export PROJECT_ROOT TRAIN_ROOT ROOT
export SEEDS="${SEEDS:-0}"
export VARIANTS="${VARIANTS:-taskonly excess_delta_p95_beta02 contextual_excess_delta_beta02 contextual_coverage_beta02 contextual_unique_beta02 contextual_adaptive_beta02}"
export EVAL_SPLITS="${EVAL_SPLITS:-clean seen holdout sparse}"
export TRAIN_STEPS="${TRAIN_STEPS:-400000}"
export TRAIN_ENVS="${TRAIN_ENVS:-16}"
export TRAIN_RATIO="${TRAIN_RATIO:-128}"
export REPLAY_SIZE="${REPLAY_SIZE:-100000}"
export BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-30000}"
export ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-30000}"
export SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-60000}"
export ADAPTIVE_TASK_TARGET="${ADAPTIVE_TASK_TARGET:-1.45}"
export MIN_FREE_GB="${MIN_FREE_GB:-80}"
export PRUNE_REPLAY_AFTER_TRAIN="${PRUNE_REPLAY_AFTER_TRAIN:-1}"
export STOP_ON_FAIL="${STOP_ON_FAIL:-1}"

"$PROJECT_ROOT/dreamerv3/run_craftax_multiseed_fault_queue.sh"
"$PROJECT_ROOT/dreamer_cuda/bin/python" \
  "$PROJECT_ROOT/dreamerv3/select_contextual_candidates.py" \
  --analysis "$ROOT/analysis" --task-retention 0.85 --top-k 2
