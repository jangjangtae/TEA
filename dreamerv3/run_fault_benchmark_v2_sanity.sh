#!/usr/bin/env bash
set -uo pipefail

# Eval-only sanity check for Crafter fault benchmark v2.
# This verifies that the v2 seen/holdout/sparse profiles produce traces and
# analysis without training a new agent.

ROOT="${ROOT:-$HOME/logdir/fault_benchmark_v2_sanity_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"
ANALYZE_PY="$PROJECT_ROOT/dreamerv3/analyze_fault_results.py"

REF_CKPT="${REF_CKPT:-/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487}"
EVAL_STEPS="${EVAL_STEPS:-20000}"
THRESH_Q="${THRESH_Q:-0.99}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
JAX_PLATFORM="${JAX_PLATFORM:-}"

mkdir -p "$ROOT"
LAUNCHER_LOG="$ROOT/launcher.log"
if [[ "${INTERNAL_LAUNCHER_LOG:-1}" == "1" && -z "${_DREAMER_QUEUE_LOGGING:-}" ]]; then
  export _DREAMER_QUEUE_LOGGING=1
  exec > >(tee -a "$LAUNCHER_LOG") 2>&1
fi
STATUS_FILE="$ROOT/status.tsv"

stamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

record_status() {
  local name="$1"
  local status="$2"
  echo -e "$(stamp)\t$name\t$status" | tee -a "$STATUS_FILE"
}

run_job() {
  local name="$1"
  shift
  local log_file="$ROOT/${name}.log"

  record_status "$name" "START"
  echo "====================================================" | tee -a "$log_file"
  echo "[$name] $(stamp)" | tee -a "$log_file"
  echo "Log: $log_file" | tee -a "$log_file"
  echo "====================================================" | tee -a "$log_file"

  "$@" 2>&1 | tee -a "$log_file"
  local code="${PIPESTATUS[0]}"
  if [[ "$code" -eq 0 ]]; then
    record_status "$name" "DONE"
  else
    record_status "$name" "FAILED:$code"
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      exit "$code"
    fi
  fi
}

run_tester_eval() {
  local name="$1"
  local splits="$2"
  local tier="$3"
  local seen_profile="$4"
  local holdout_profile="$5"
  local outdir="$ROOT/$name"
  local jax_args=()
  if [[ -n "$JAX_PLATFORM" ]]; then
    jax_args=(--jax.platform "$JAX_PLATFORM")
  fi
  mkdir -p "$outdir"
  unset CRAFTER_OUTPUT_DIR || true
  unset CRAFTER_TRACE_PATH || true

  TESTER_EVAL_CHECKPOINT="$REF_CKPT" \
  TESTER_REF_CHECKPOINT="$REF_CKPT" \
  TESTER_EVAL_STEPS="$EVAL_STEPS" \
  TESTER_EVAL_THRESHOLD_Q="$THRESH_Q" \
  TESTER_EVAL_SPLITS="$splits" \
  TESTER_EVAL_FAULT_FREQ_TIER="$tier" \
  TESTER_EVAL_SEEN_FAULT_PROFILE="$seen_profile" \
  TESTER_EVAL_HOLDOUT_FAULT_PROFILE="$holdout_profile" \
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
    --script tester_eval \
    --configs crafter \
    --logdir "$outdir" \
    --run.from_checkpoint "$REF_CKPT" \
    --fault.enabled True \
    --fault.ref_ckpt "$REF_CKPT" \
    --fault.use_reward_error True \
    --fault.w_reward 0.0 \
    "${jax_args[@]}"
}

run_analysis() {
  local outdir="$ROOT/analysis"
  run_job "99_analysis" \
    "$PYTHON_BIN" "$ANALYZE_PY" \
      --roots "$ROOT" \
      --outdir "$outdir" \
      --splits clean,seen,holdout \
      --trace-analysis \
      --top-fracs 0.01,0.05
}

echo -e "time\tname\tstatus" > "$STATUS_FILE"

echo "===================================================="
echo "[Crafter fault benchmark v2 sanity]"
echo "root          : $ROOT"
echo "reference ckpt: $REF_CKPT"
echo "eval steps    : $EVAL_STEPS"
echo "threshold q   : $THRESH_Q"
echo "jax platform  : ${JAX_PLATFORM:-config default}"
echo "===================================================="

run_job "01_v2_benchmark_eval" \
  run_tester_eval "01_v2_benchmark_eval" "clean,seen,holdout" \
    "benchmark" "benchmark_v2_seen" "benchmark_v2_holdout"

run_job "02_v2_sparse_holdout_eval" \
  run_tester_eval "02_v2_sparse_holdout_eval" "clean,holdout" \
    "realistic" "benchmark_v2_seen" "benchmark_v2_sparse"

run_analysis

record_status "fault_benchmark_v2_sanity" "FINISHED"
echo "Saved under: $ROOT"
