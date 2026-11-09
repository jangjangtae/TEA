#!/usr/bin/env bash
set -euo pipefail

# End-to-end rehearsal of the exact vacation wrapper with tiny budgets. This
# exercises all selected objectives, milestone resume/archive/evaluation,
# aggregate plots, candidate selection, and the detector replication phase.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
ROOT="${ROOT:-/tmp/craftax_paper_crl_vacation_smoke_$(date +%Y%m%d_%H%M%S)}"
cd "$PROJECT_ROOT"

echo "Craftax vacation smoke root: $ROOT"

ROOT="$ROOT" \
ADAPT_SEEDS="0" \
ADAPT_VARIANTS="taskonly excess_delta_p95_beta02 contextual_excess_delta_beta02 klbound_crl_task85 contextual_crl_task85" \
TRAIN_STEPS="512" \
TRAIN_MILESTONES="256 512" \
TRAIN_ENVS="1" \
TRAIN_RATIO="1" \
REPLAY_SIZE="1000" \
BATCH_SIZE="2" \
BATCH_LENGTH="8" \
REPORT_LENGTH="8" \
BASE_EVAL_STEPS="128" \
ADAPT_EVAL_STEPS="128" \
SPARSE_EVAL_STEPS="192" \
MILESTONE_EVAL_STEPS="128" \
MILESTONE_SPARSE_EVAL_STEPS="192" \
DETECT_SEEDS="0" \
DETECT_STEPS="128" \
DETECT_SPARSE_STEPS="192" \
MIN_FREE_GB="5" \
STOP_ON_PHASE_FAIL="1" \
  "$PROJECT_ROOT/dreamerv3/run_craftax_paper_crl_vacation.sh"

echo "Craftax vacation smoke DONE: $ROOT"
