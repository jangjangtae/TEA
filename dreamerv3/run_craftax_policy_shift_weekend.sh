#!/usr/bin/env bash
set -euo pipefail

# Longer Craftax adaptation queue for checking whether the policy actually
# shifts toward tester-like behavior under fault-score objectives.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-$HOME/logdir/craftax_policy_shift_weekend_$(date +%Y%m%d_%H%M%S)}"

export PROJECT_ROOT
export TRAIN_ROOT
export ROOT

export SEEDS="${SEEDS:-0 1 2}"
export VARIANTS="${VARIANTS:-taskonly dense_beta02 excess_p95_beta02 excess_delta_p95_beta02}"
export EVAL_SPLITS="${EVAL_SPLITS:-clean seen holdout sparse}"

export TRAIN_STEPS="${TRAIN_STEPS:-800000}"
export TRAIN_ENVS="${TRAIN_ENVS:-16}"
export TRAIN_RATIO="${TRAIN_RATIO:-128}"
export REPLAY_SIZE="${REPLAY_SIZE:-150000}"
export SAVE_EVERY="${SAVE_EVERY:-600}"

export BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-50000}"
export ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-50000}"
export SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-50000}"
export EVAL_ENVS="${EVAL_ENVS:-1}"

export BATCH_SIZE="${BATCH_SIZE:-16}"
export BATCH_LENGTH="${BATCH_LENGTH:-64}"
export REPORT_LENGTH="${REPORT_LENGTH:-32}"

export JAX_PLATFORM="${JAX_PLATFORM:-cuda}"
export JAX_PREALLOC="${JAX_PREALLOC:-True}"
export JAX_COMPUTE_DTYPE="${JAX_COMPUTE_DTYPE:-float32}"

export MIN_FREE_GB="${MIN_FREE_GB:-80}"
export PRUNE_REPLAY_AFTER_TRAIN="${PRUNE_REPLAY_AFTER_TRAIN:-1}"
export STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
export RUN_BASE_EVALS="${RUN_BASE_EVALS:-1}"
export RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

exec "$PROJECT_ROOT/dreamerv3/run_craftax_multiseed_fault_queue.sh"
