#!/usr/bin/env bash
set -euo pipefail

# One-day methodological probe:
# 1) ICML 2025 KL-bound detector vs posterior-prior KL, evaluation only.
# 2) Fixed-beta KL-bound reward vs a task-constrained primal-dual variant.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-$HOME/logdir/craftax_paper_crl_oneday_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

echo "Paper/CRL one-day root: $ROOT"

DETECT_ROOT="$ROOT/01_kl_bound_detection" \
ROOT="$ROOT/01_kl_bound_detection" \
TRAIN_ROOT="$TRAIN_ROOT" SEEDS="${DETECT_SEEDS:-0}" \
EVAL_STEPS="${DETECT_STEPS:-20000}" SPARSE_STEPS="${DETECT_SPARSE_STEPS:-40000}" \
  "$PROJECT_ROOT/dreamerv3/run_craftax_kl_bound_probe.sh"

ADAPT_ROOT="$ROOT/02_adaptation"
ROOT="$ADAPT_ROOT" TRAIN_ROOT="$TRAIN_ROOT" \
SEEDS="${ADAPT_SEEDS:-0}" \
VARIANTS="${ADAPT_VARIANTS:-taskonly contextual_excess_delta_beta02 klbound_reward_beta02 klbound_crl_task85}" \
EVAL_SPLITS="clean seen holdout sparse" \
TRAIN_STEPS="${TRAIN_STEPS:-300000}" TRAIN_ENVS="16" TRAIN_RATIO="128" \
REPLAY_SIZE="100000" BASE_EVAL_STEPS="30000" ADAPT_EVAL_STEPS="30000" \
SPARSE_EVAL_STEPS="60000" CONSTRAINT_TASK_TARGET="${CONSTRAINT_TASK_TARGET:-1.45}" \
MIN_FREE_GB="80" PRUNE_REPLAY_AFTER_TRAIN="1" STOP_ON_FAIL="1" \
  "$PROJECT_ROOT/dreamerv3/run_craftax_multiseed_fault_queue.sh"

"$PROJECT_ROOT/dreamer_cuda/bin/python" \
  "$PROJECT_ROOT/dreamerv3/select_contextual_candidates.py" \
  --analysis "$ADAPT_ROOT/analysis" --task-retention 0.85 --top-k 2

echo "Paper/CRL one-day probe DONE: $ROOT"
