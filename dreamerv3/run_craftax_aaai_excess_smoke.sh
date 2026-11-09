#!/usr/bin/env bash
set -euo pipefail

# Tiny end-to-end rehearsal for the AAAI excess-novelty ablation queue.
# This should be run before unattended jobs; it exercises clean calibration,
# contextual normalization, milestone checkpoints, eval traces, and analysis.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-/tmp/craftax_aaai_excess_smoke_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

echo "Craftax AAAI excess smoke root: $ROOT"

ROOT="$ROOT" \
TRAIN_ROOT="$TRAIN_ROOT" \
SEEDS="${SEEDS:-0}" \
VARIANTS="${VARIANTS:-taskonly dense_beta02 excess_delta_p95_beta02 contextual_excess_delta_beta02}" \
EVAL_SPLITS="${EVAL_SPLITS:-clean seen sparse}" \
TRAIN_STEPS="${TRAIN_STEPS:-512}" \
TRAIN_MILESTONES="${TRAIN_MILESTONES:-256 512}" \
TRAIN_ENVS="${TRAIN_ENVS:-1}" \
TRAIN_RATIO="${TRAIN_RATIO:-1}" \
REPLAY_SIZE="${REPLAY_SIZE:-1000}" \
BATCH_SIZE="${BATCH_SIZE:-2}" \
BATCH_LENGTH="${BATCH_LENGTH:-8}" \
REPORT_LENGTH="${REPORT_LENGTH:-8}" \
BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-128}" \
ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-128}" \
SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-192}" \
MILESTONE_EVAL_STEPS="${MILESTONE_EVAL_STEPS:-128}" \
MILESTONE_SPARSE_EVAL_STEPS="${MILESTONE_SPARSE_EVAL_STEPS:-192}" \
MIN_FREE_GB="${MIN_FREE_GB:-5}" \
STOP_ON_FAIL="${STOP_ON_FAIL:-1}" \
RUN_ANALYSIS="${RUN_ANALYSIS:-1}" \
PRUNE_REPLAY_AFTER_TRAIN="${PRUNE_REPLAY_AFTER_TRAIN:-1}" \
  "$PROJECT_ROOT/dreamerv3/run_craftax_multiseed_fault_queue.sh"

echo "Craftax AAAI excess smoke DONE: $ROOT"
