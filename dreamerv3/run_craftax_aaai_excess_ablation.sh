#!/usr/bin/env bash
set -euo pipefail

# AAAI-focused ablation around the current strongest method:
# contextual excess-delta fault reward. The variant set is intentionally small:
# task-only, naive dense shaping, global excess-delta, and the contextual main
# method. Override VARIANTS to add beta or quantile sensitivity after the core
# result is secured.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-$HOME/logdir/craftax_aaai_excess_ablation_$(date +%Y%m%d_%H%M%S)}"

export PROJECT_ROOT TRAIN_ROOT ROOT
export SEEDS="${SEEDS:-0 1 2}"
export VARIANTS="${VARIANTS:-taskonly dense_beta02 excess_delta_p95_beta02 contextual_excess_delta_beta02}"
export EVAL_SPLITS="${EVAL_SPLITS:-clean seen holdout sparse}"

# Final AAAI budget. Earlier pilots suggested that 200k adaptation can be too
# short to visibly change the tester policy, while 800k-1M is where sparse bug
# discovery becomes more informative. Milestones let us report whether the
# effect appears early or only after extended adaptation.
export TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
export TRAIN_MILESTONES="${TRAIN_MILESTONES:-200000 400000 600000 800000 1000000}"
export TRAIN_ENVS="${TRAIN_ENVS:-16}"
export TRAIN_RATIO="${TRAIN_RATIO:-128}"
export REPLAY_SIZE="${REPLAY_SIZE:-100000}"

export BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-30000}"
export ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-30000}"
export SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-60000}"
export MILESTONE_EVAL_STEPS="${MILESTONE_EVAL_STEPS:-10000}"
export MILESTONE_SPARSE_EVAL_STEPS="${MILESTONE_SPARSE_EVAL_STEPS:-20000}"

export MIN_FREE_GB="${MIN_FREE_GB:-35}"
export PRUNE_REPLAY_AFTER_TRAIN="${PRUNE_REPLAY_AFTER_TRAIN:-1}"
export STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
export RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

exec "$PROJECT_ROOT/dreamerv3/run_craftax_multiseed_fault_queue.sh"
