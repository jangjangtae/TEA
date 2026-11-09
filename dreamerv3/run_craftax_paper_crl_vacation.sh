#!/usr/bin/env bash
set -uo pipefail

# Vacation queue sized from the measured 70-hour, 3-seed, 4-variant 800K run.
# Phase 1 confirms five objectives over five seeds. Phase 2 spends the remaining
# window on a longer multi-seed detector comparison. A phase failure is logged
# and does not discard the other phase by default.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-$HOME/logdir/craftax_paper_crl_vacation_$(date +%Y%m%d_%H%M%S)}"
STATUS="$ROOT/phase_status.tsv"
STOP_ON_PHASE_FAIL="${STOP_ON_PHASE_FAIL:-0}"
cd "$PROJECT_ROOT"
mkdir -p "$ROOT"

record() {
  printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" | tee -a "$STATUS"
}

run_adaptation() {
  ROOT="$ROOT/01_adaptation_confirmation" \
  TRAIN_ROOT="$TRAIN_ROOT" \
  SEEDS="${ADAPT_SEEDS:-0 1 2 3 4}" \
  VARIANTS="${ADAPT_VARIANTS:-taskonly excess_delta_p95_beta02 contextual_excess_delta_beta02 klbound_crl_task85 contextual_crl_task85}" \
  TRAIN_STEPS="${TRAIN_STEPS:-1000000}" \
  TRAIN_MILESTONES="${TRAIN_MILESTONES:-200000 400000 600000 800000 1000000}" \
  BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-50000}" \
  ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-50000}" \
  SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-100000}" \
  MILESTONE_EVAL_STEPS="${MILESTONE_EVAL_STEPS:-15000}" \
  MILESTONE_SPARSE_EVAL_STEPS="${MILESTONE_SPARSE_EVAL_STEPS:-30000}" \
  MIN_FREE_GB="${MIN_FREE_GB:-55}" \
    "$PROJECT_ROOT/dreamerv3/run_craftax_paper_crl_weeklong.sh"
}

run_detector() {
  ROOT="$ROOT/02_kl_bound_detector_replication" \
  TRAIN_ROOT="$TRAIN_ROOT" \
  SEEDS="${DETECT_SEEDS:-0 1 2}" \
  EVAL_STEPS="${DETECT_STEPS:-60000}" \
  SPARSE_STEPS="${DETECT_SPARSE_STEPS:-120000}" \
  MIN_FREE_GB="${MIN_FREE_GB:-55}" \
    "$PROJECT_ROOT/dreamerv3/run_craftax_kl_bound_probe.sh"
}

run_phase() {
  local name="$1"
  shift
  record "$name" START
  "$@"
  local code=$?
  if (( code == 0 )); then
    record "$name" DONE
  else
    record "$name" "FAIL($code)"
    if [[ "$STOP_ON_PHASE_FAIL" == "1" ]]; then
      return "$code"
    fi
  fi
  return 0
}

echo "===================================================="
echo "[Craftax paper/CRL vacation queue]"
echo "root             : $ROOT"
echo "adapt seeds      : ${ADAPT_SEEDS:-0 1 2 3 4}"
echo "adapt variants   : ${ADAPT_VARIANTS:-taskonly excess_delta_p95_beta02 contextual_excess_delta_beta02 klbound_crl_task85 contextual_crl_task85}"
echo "adapt steps      : ${TRAIN_STEPS:-1000000}"
echo "adapt milestones : ${TRAIN_MILESTONES:-200000 400000 600000 800000 1000000}"
echo "interim eval     : ${MILESTONE_EVAL_STEPS:-15000} / sparse ${MILESTONE_SPARSE_EVAL_STEPS:-30000}"
echo "final eval       : ${ADAPT_EVAL_STEPS:-50000} / sparse ${SPARSE_EVAL_STEPS:-100000}"
echo "detector seeds   : ${DETECT_SEEDS:-0 1 2}"
echo "detector steps   : ${DETECT_STEPS:-60000} / sparse ${DETECT_SPARSE_STEPS:-120000}"
echo "continue on fail : $((1 - STOP_ON_PHASE_FAIL))"
echo "===================================================="

run_phase adaptation_confirmation run_adaptation || exit $?
run_phase kl_bound_detector_replication run_detector || exit $?

record vacation_queue DONE
echo "Vacation queue DONE: $ROOT"
