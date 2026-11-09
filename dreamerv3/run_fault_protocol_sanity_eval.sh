#!/usr/bin/env bash
set -uo pipefail

# Short eval-only protocol check for the revised Crafter fault labels.
# This verifies four cases before launching expensive runs:
#   1) clean false-alarm baseline
#   2) forced semantic manifestation
#   3) trigger context without manifestation
#   4) benchmark-frequency stochastic semantic faults

ROOT="${ROOT:-$HOME/logdir/fault_protocol_sanity_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"
PATCH_PY="$PROJECT_ROOT/dreamerv3/patch_crafter_semantic_manifest.py"

REF_CKPT="${REF_CKPT:-/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487}"
EVAL_STEPS="${EVAL_STEPS:-10000}"
THRESH_Q="${THRESH_Q:-0.99}"
RESUME_EXISTING="${RESUME_EXISTING:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
JAX_PLATFORM="${JAX_PLATFORM:-}"

SEMANTIC_HOLDOUT_SUBTYPES="${SEMANTIC_HOLDOUT_SUBTYPES:-tool_collect_desync_on_upgrade,craft_result_missing_on_retry,station_place_ghost_on_relocate,achievement_unlock_missing_after_valid_progress,station_usable_flag_broken_after_relocate,recipe_precondition_mischeck_on_retry,delayed_inventory_desync_after_station_use}"

mkdir -p "$ROOT"
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
  local fault_tier="$3"
  local semantic_ep_prob="$4"
  local manifest_prob="$5"
  local semantic_profile="${6:-eval_holdout}"
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
  TESTER_EVAL_RESUME_EXISTING="$RESUME_EXISTING" \
  TESTER_EVAL_FAULT_FREQ_TIER="$fault_tier" \
  TESTER_EVAL_SEMANTIC_FAULT_PROFILE="$semantic_profile" \
  TESTER_EVAL_SEMANTIC_FAULT_EP_PROB="$semantic_ep_prob" \
  TESTER_EVAL_SEMANTIC_FAULT_MANIFEST_PROB="$manifest_prob" \
  TESTER_EVAL_SEMANTIC_SUBTYPES="$SEMANTIC_HOLDOUT_SUBTYPES" \
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
    "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/analyze_fault_results.py" \
      --roots "$ROOT" \
      --outdir "$outdir" \
      --trace-analysis \
      --splits clean,semantic_holdout \
      --top-fracs 0.01,0.05
}

echo -e "time\tname\tstatus" > "$STATUS_FILE"

echo "===================================================="
echo "[Fault protocol sanity eval]"
echo "root          : $ROOT"
echo "reference ckpt: $REF_CKPT"
echo "eval steps    : $EVAL_STEPS"
echo "threshold q   : $THRESH_Q"
echo "jax platform  : ${JAX_PLATFORM:-config default}"
echo "resume existing: $RESUME_EXISTING"
echo "semantic types: $SEMANTIC_HOLDOUT_SUBTYPES"
echo "===================================================="

if [[ -x "$PYTHON_BIN" && -f "$PATCH_PY" ]]; then
  run_job "00_patch_crafter_semantic_manifest" "$PYTHON_BIN" "$PATCH_PY"
fi

run_job "01_clean_reference" \
  run_tester_eval "01_clean_reference" "clean" "custom" "0.0" "0.0"

run_job "02_semantic_forced_manifest" \
  run_tester_eval "02_semantic_forced_manifest" "clean,semantic_holdout" \
    "diagnostic" "1.0" "1.0"

run_job "03_semantic_trigger_only" \
  run_tester_eval "03_semantic_trigger_only" "clean,semantic_holdout" \
    "diagnostic" "1.0" "0.0"

run_job "04_semantic_benchmark_stochastic" \
  run_tester_eval "04_semantic_benchmark_stochastic" "clean,semantic_holdout" \
    "benchmark" "0.25" "0.5"

run_analysis

record_status "fault_protocol_sanity_eval" "FINISHED"
echo "Saved under: $ROOT"
