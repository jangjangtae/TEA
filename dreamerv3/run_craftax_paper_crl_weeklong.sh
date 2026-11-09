#!/usr/bin/env bash
set -euo pipefail

# Week-long confirmation queue. Run this only after inspecting the one-day
# screen; override VARIANTS to exclude methods that fail its decision gates.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-$HOME/logdir/craftax_paper_crl_weeklong_$(date +%Y%m%d_%H%M%S)}"
cd "$PROJECT_ROOT"
mkdir -p "$ROOT"

echo "===================================================="
echo "[Craftax paper/CRL week-long confirmation]"
echo "root       : $ROOT"
echo "seeds      : ${SEEDS:-0 1 2 3 4}"
echo "variants   : ${VARIANTS:-taskonly excess_delta_p95_beta02 contextual_excess_delta_beta02 klbound_crl_task85 contextual_crl_task85}"
echo "train steps: ${TRAIN_STEPS:-1000000}"
echo "milestones : ${TRAIN_MILESTONES:-200000 400000 600000 800000 1000000}"
echo "===================================================="

ROOT="$ROOT" TRAIN_ROOT="$TRAIN_ROOT" \
SEEDS="${SEEDS:-0 1 2 3 4}" \
VARIANTS="${VARIANTS:-taskonly excess_delta_p95_beta02 contextual_excess_delta_beta02 klbound_crl_task85 contextual_crl_task85}" \
EVAL_SPLITS="${EVAL_SPLITS:-clean seen holdout sparse}" \
TRAIN_STEPS="${TRAIN_STEPS:-1000000}" \
TRAIN_MILESTONES="${TRAIN_MILESTONES:-200000 400000 600000 800000 1000000}" \
TRAIN_ENVS="${TRAIN_ENVS:-16}" TRAIN_RATIO="${TRAIN_RATIO:-128}" \
REPLAY_SIZE="${REPLAY_SIZE:-100000}" \
BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-50000}" \
ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-50000}" \
SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-100000}" \
MILESTONE_EVAL_STEPS="${MILESTONE_EVAL_STEPS:-15000}" \
MILESTONE_SPARSE_EVAL_STEPS="${MILESTONE_SPARSE_EVAL_STEPS:-30000}" \
CONSTRAINT_TASK_TARGET="${CONSTRAINT_TASK_TARGET:-1.45}" \
CONSTRAINT_WARMUP_EPISODES="${CONSTRAINT_WARMUP_EPISODES:-10}" \
MIN_FREE_GB="${MIN_FREE_GB:-55}" \
PRUNE_REPLAY_AFTER_TRAIN="1" STOP_ON_FAIL="1" \
  "$PROJECT_ROOT/dreamerv3/run_craftax_multiseed_fault_queue.sh"

FINAL_MILESTONE="${TRAIN_MILESTONES:-200000 400000 600000 800000 1000000}"
FINAL_MILESTONE="${FINAL_MILESTONE##* }"
"$PROJECT_ROOT/dreamer_cuda/bin/python" \
  "$PROJECT_ROOT/dreamerv3/select_contextual_candidates.py" \
  --analysis "$ROOT/analysis/milestone_${FINAL_MILESTONE}" \
  --task-retention 0.85 --top-k 3

echo "Week-long confirmation DONE: $ROOT"
