#!/usr/bin/env bash
set -euo pipefail

# Follow-up experiments for the AAAI FATE revision.
#
# This script is intentionally staged after the main seed 3/4 completion run.
# It does not modify or resume the main result directory. It writes a separate
# root containing:
#   1) KL-bound and FATE component ablations.
#   2) Sparse-split sensitivity checks for percentile and beta.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-$HOME/logdir/craftax_aaai_followup_$(date +%Y%m%d_%H%M%S)}"
WAIT_SERVICE="${WAIT_SERVICE:-craftax-seed34-all-methods-20260715.service}"

SEEDS_COMPONENT="${SEEDS_COMPONENT:-0 1 2}"
SEEDS_SENSITIVITY="${SEEDS_SENSITIVITY:-0 1 2}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
TRAIN_ENVS="${TRAIN_ENVS:-16}"
TRAIN_RATIO="${TRAIN_RATIO:-128}"
REPLAY_SIZE="${REPLAY_SIZE:-100000}"
MIN_FREE_GB="${MIN_FREE_GB:-35}"

BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-30000}"
ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-30000}"
SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-60000}"
MILESTONE_EVAL_STEPS="${MILESTONE_EVAL_STEPS:-10000}"
MILESTONE_SPARSE_EVAL_STEPS="${MILESTONE_SPARSE_EVAL_STEPS:-20000}"

RUN_COMPONENT="${RUN_COMPONENT:-1}"
RUN_SENSITIVITY="${RUN_SENSITIVITY:-1}"

cd "$PROJECT_ROOT"
mkdir -p "$ROOT"
LAUNCHER_LOG="$ROOT/launcher.log"
STATUS_FILE="$ROOT/status.tsv"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

stamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

wait_for_service() {
  local service="$1"
  if [[ -z "$service" ]]; then
    return
  fi
  echo "[$(stamp)] waiting for service: $service"
  while true; do
    if systemctl --user is-active --quiet "$service" 2>/dev/null; then
      systemctl --user status "$service" --no-pager | sed -n '1,12p' || true
      sleep 600
      continue
    fi
    # Fallback for shells without access to the user systemd bus.
    if pgrep -f "run_craftax_seed34_all_methods.sh" >/dev/null 2>&1; then
      echo "[$(stamp)] seed34 queue still found by pgrep; waiting"
      sleep 600
      continue
    fi
    break
  done
  echo "[$(stamp)] wait complete: $service is no longer active"
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
echo "[Craftax AAAI follow-up queue]"
echo "root          : $ROOT"
echo "train root    : $TRAIN_ROOT"
echo "wait service  : ${WAIT_SERVICE:-<none>}"
echo "component     : $RUN_COMPONENT seeds=[$SEEDS_COMPONENT]"
echo "sensitivity   : $RUN_SENSITIVITY seeds=[$SEEDS_SENSITIVITY]"
echo "train steps   : $TRAIN_STEPS"
echo "replay        : $REPLAY_SIZE, prune after train"
echo "min free GB   : $MIN_FREE_GB"
echo "===================================================="

wait_for_service "$WAIT_SERVICE"

if [[ "$RUN_COMPONENT" == "1" ]]; then
  run_stage "01_component_klbound" env \
    ROOT="$ROOT/01_component_klbound" \
    TRAIN_ROOT="$TRAIN_ROOT" \
    SEEDS="$SEEDS_COMPONENT" \
    VARIANTS="klbound_reward_beta02 excess_p95_beta02 delta_p95_beta02" \
    EVAL_SPLITS="seen holdout sparse" \
    TRAIN_STEPS="$TRAIN_STEPS" \
    TRAIN_MILESTONES="$TRAIN_STEPS" \
    TRAIN_ENVS="$TRAIN_ENVS" \
    TRAIN_RATIO="$TRAIN_RATIO" \
    REPLAY_SIZE="$REPLAY_SIZE" \
    RUN_BASE_EVALS=0 \
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
fi

if [[ "$RUN_SENSITIVITY" == "1" ]]; then
  run_stage "02_sparse_sensitivity" env \
    ROOT="$ROOT/02_sparse_sensitivity" \
    TRAIN_ROOT="$TRAIN_ROOT" \
    SEEDS="$SEEDS_SENSITIVITY" \
    VARIANTS="excess_delta_p90_beta02 excess_delta_p99_beta02 excess_delta_p95_beta01 excess_delta_p95_beta04" \
    EVAL_SPLITS="sparse" \
    TRAIN_STEPS="$TRAIN_STEPS" \
    TRAIN_MILESTONES="$TRAIN_STEPS" \
    TRAIN_ENVS="$TRAIN_ENVS" \
    TRAIN_RATIO="$TRAIN_RATIO" \
    REPLAY_SIZE="$REPLAY_SIZE" \
    RUN_BASE_EVALS=0 \
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
fi

echo "===================================================="
echo "[Craftax AAAI follow-up queue] DONE"
echo "root   : $ROOT"
echo "status : $STATUS_FILE"
echo "log    : $LAUNCHER_LOG"
echo "===================================================="
