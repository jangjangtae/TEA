#!/usr/bin/env bash
set -uo pipefail

# Eval-only queue for stricter held-out bug analysis.
# It re-evaluates representative checkpoints with:
#   clean, seen, holdout, semantic_holdout

ROOT="${ROOT:-$HOME/logdir/fault_semantic_holdout_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"

REF_CKPT="${REF_CKPT:-/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487}"
FOLLOWUP_ROOT="${FOLLOWUP_ROOT:-/home/railab/logdir/fault_followup3_20260604_175648}"
EVAL_STEPS="${EVAL_STEPS:-300000}"
THRESH_Q="${THRESH_Q:-0.99}"
EVAL_SPLITS="${EVAL_SPLITS:-clean,seen,holdout,semantic_holdout}"
RESUME_EXISTING="${RESUME_EXISTING:-0}"
SEMANTIC_MANIFEST_PROB="${SEMANTIC_MANIFEST_PROB:-${TESTER_EVAL_SEMANTIC_FAULT_MANIFEST_PROB:-1.0}}"

SEMANTIC_SUBTYPES="${SEMANTIC_SUBTYPES:-tool_collect_desync_on_upgrade,craft_result_missing_on_retry,station_place_ghost_on_relocate,achievement_unlock_missing_after_valid_progress,station_usable_flag_broken_after_relocate,recipe_precondition_mischeck_on_retry,delayed_inventory_desync_after_station_use}"

RUN_REFERENCE="${RUN_REFERENCE:-1}"
RUN_BETA005="${RUN_BETA005:-1}"
RUN_BETA01_REPEAT="${RUN_BETA01_REPEAT:-1}"
RUN_TASK_ONLY_REPEAT="${RUN_TASK_ONLY_REPEAT:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"

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

latest_ckpt_dir() {
  local ckpt_root="$1/ckpt"
  if [[ -f "$ckpt_root/latest" ]]; then
    local latest
    latest="$(cat "$ckpt_root/latest")"
    if [[ -d "$ckpt_root/$latest" ]]; then
      echo "$ckpt_root/$latest"
      return 0
    fi
  fi
  find "$ckpt_root" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
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
  return 0
}

run_tester_eval() {
  local outname="$1"
  local ckpt="$2"
  local outdir="$ROOT/$outname"
  mkdir -p "$outdir"

  unset CRAFTER_OUTPUT_DIR || true
  unset CRAFTER_TRACE_PATH || true

  TESTER_EVAL_CHECKPOINT="$ckpt" \
  TESTER_REF_CHECKPOINT="$REF_CKPT" \
  TESTER_EVAL_STEPS="$EVAL_STEPS" \
  TESTER_EVAL_THRESHOLD_Q="$THRESH_Q" \
  TESTER_EVAL_SPLITS="$EVAL_SPLITS" \
  TESTER_EVAL_RESUME_EXISTING="$RESUME_EXISTING" \
  TESTER_EVAL_SEMANTIC_FAULT_EP_PROB="${TESTER_EVAL_SEMANTIC_FAULT_EP_PROB:-0.5}" \
  TESTER_EVAL_SEMANTIC_FAULT_MANIFEST_PROB="$SEMANTIC_MANIFEST_PROB" \
  TESTER_EVAL_SEMANTIC_SUBTYPES="$SEMANTIC_SUBTYPES" \
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
    --script tester_eval \
    --configs crafter \
    --logdir "$outdir" \
    --run.from_checkpoint "$ckpt" \
    --fault.enabled True \
    --fault.ref_ckpt "$REF_CKPT" \
    --fault.use_reward_error True \
    --fault.w_reward 0.0
}

echo -e "time\tname\tstatus" > "$STATUS_FILE"

CKPT_BETA005="$(latest_ckpt_dir "$FOLLOWUP_ROOT/fault_adapt_beta005")"
CKPT_BETA01_REPEAT="$(latest_ckpt_dir "$FOLLOWUP_ROOT/fault_adapt_beta01_repeat")"
CKPT_TASK_ONLY_REPEAT="$(latest_ckpt_dir "$FOLLOWUP_ROOT/task_only_fault_logging_repeat")"

echo "===================================================="
echo "[Semantic holdout eval queue]"
echo "root          : $ROOT"
echo "reference ckpt: $REF_CKPT"
echo "followup root : $FOLLOWUP_ROOT"
echo "eval steps    : $EVAL_STEPS"
echo "splits        : $EVAL_SPLITS"
echo "resume existing: $RESUME_EXISTING"
echo "semantic manifest prob: $SEMANTIC_MANIFEST_PROB"
echo "semantic types: $SEMANTIC_SUBTYPES"
echo "stop on fail : $STOP_ON_FAIL"
echo "===================================================="

if [[ "$RUN_REFERENCE" == "1" ]]; then
  run_job "01_reference_semantic_eval" run_tester_eval "reference_semantic_eval" "$REF_CKPT"
fi

if [[ "$RUN_BETA005" == "1" ]]; then
  if [[ -n "${CKPT_BETA005:-}" && -d "$CKPT_BETA005" ]]; then
    run_job "02_beta005_semantic_eval" run_tester_eval "beta005_semantic_eval" "$CKPT_BETA005"
  else
    record_status "02_beta005_semantic_eval" "SKIPPED:no_ckpt"
  fi
fi

if [[ "$RUN_BETA01_REPEAT" == "1" ]]; then
  if [[ -n "${CKPT_BETA01_REPEAT:-}" && -d "$CKPT_BETA01_REPEAT" ]]; then
    run_job "03_beta01_repeat_semantic_eval" run_tester_eval "beta01_repeat_semantic_eval" "$CKPT_BETA01_REPEAT"
  else
    record_status "03_beta01_repeat_semantic_eval" "SKIPPED:no_ckpt"
  fi
fi

if [[ "$RUN_TASK_ONLY_REPEAT" == "1" ]]; then
  if [[ -n "${CKPT_TASK_ONLY_REPEAT:-}" && -d "$CKPT_TASK_ONLY_REPEAT" ]]; then
    run_job "04_task_only_repeat_semantic_eval" run_tester_eval "task_only_repeat_semantic_eval" "$CKPT_TASK_ONLY_REPEAT"
  else
    record_status "04_task_only_repeat_semantic_eval" "SKIPPED:no_ckpt"
  fi
fi

record_status "semantic_holdout_eval_queue" "FINISHED"
echo "Saved under: $ROOT"
