#!/usr/bin/env bash
set -uo pipefail

# Resume only the unfinished long tester-eval jobs from the weekend queue.
# This intentionally avoids the nested block runner in run_fault_weekend_queue.sh
# so interrupted long evals can be restarted with clear status entries.

ROOT="${ROOT:-/home/railab/logdir/fault_weekend_20260612_134024}"
CURRENT_ROOT="${CURRENT_ROOT:-/home/railab/logdir/fault_semantic_full_20260610_142128}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"

REF_CKPT="${REF_CKPT:-/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487}"
LONG_EVAL_STEPS="${LONG_EVAL_STEPS:-300000}"
LONG_EVAL_SPLITS="${LONG_EVAL_SPLITS:-clean,seen,holdout,semantic_holdout}"
THRESH_Q="${THRESH_Q:-0.99}"
SEMANTIC_EVAL_EP_PROB="${SEMANTIC_EVAL_EP_PROB:-0.5}"
CASES="${CASES:-ungated_beta005,oracle_tester}"
RUN_FINAL_ANALYSIS="${RUN_FINAL_ANALYSIS:-1}"
STATUS_FILE="${STATUS_FILE:-$ROOT/status.tsv}"

SEMANTIC_HOLDOUT_SUBTYPES="${SEMANTIC_HOLDOUT_SUBTYPES:-tool_collect_desync_on_upgrade,craft_result_missing_on_retry,station_place_ghost_on_relocate,achievement_unlock_missing_after_valid_progress,station_usable_flag_broken_after_relocate,recipe_precondition_mischeck_on_retry,delayed_inventory_desync_after_station_use}"

PREV_ROOTS=(
  /home/railab/logdir/fault_3day_leave_20260528_170600
  /home/railab/logdir/fault_nextday_20260602_182008
  /home/railab/logdir/fault_followup3_20260604_175648
  /home/railab/logdir/fault_semantic_holdout_20260608_134437_systemd
  /home/railab/logdir/fault_protocol_eval_20260609_145011
  /home/railab/logdir/fault_protocol_night_followup_20260609_174107
)

CONTEXT_ROOTS=(
  /home/railab/logdir/fault_protocol_eval_20260609_145011
  /home/railab/logdir/fault_protocol_night_followup_20260609_174107
)

mkdir -p "$ROOT"
if [[ ! -f "$STATUS_FILE" ]]; then
  echo -e "time\tname\tstatus" > "$STATUS_FILE"
fi

stamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

record_status() {
  local name="$1"
  local status="$2"
  echo -e "$(stamp)\t$name\t$status" | tee -a "$STATUS_FILE"
}

as_words() {
  echo "${1//,/ }"
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

current_case_ckpt() {
  local case_name="$1"
  case "$case_name" in
    ungated_beta005)
      latest_ckpt_dir "$CURRENT_ROOT/semantic_fault_ungated_beta005"
      ;;
    oracle_tester)
      latest_ckpt_dir "$CURRENT_ROOT/semantic_oracle_tester_reward"
      ;;
    task_only)
      latest_ckpt_dir "$CURRENT_ROOT/semantic_task_only_fault_logging"
      ;;
    gated_beta005)
      latest_ckpt_dir "$CURRENT_ROOT/semantic_fault_gated_beta005"
      ;;
    gated_beta01)
      latest_ckpt_dir "$CURRENT_ROOT/semantic_fault_gated_beta01"
      ;;
    reference)
      echo "$REF_CKPT"
      ;;
    *)
      echo ""
      ;;
  esac
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
    exit "$code"
  fi
}

run_tester_eval() {
  local outname="$1"
  local ckpt="$2"
  local outdir="$ROOT/$outname"
  mkdir -p "$outdir"

  unset CRAFTER_OUTPUT_DIR || true
  unset CRAFTER_TRACE_PATH || true

  echo "Tester eval outdir: $outdir"
  echo "Tester checkpoint : $ckpt"
  echo "Reference ckpt    : $REF_CKPT"
  echo "Splits            : $LONG_EVAL_SPLITS"
  echo "Steps per split   : $LONG_EVAL_STEPS"

  TESTER_EVAL_CHECKPOINT="$ckpt" \
  TESTER_REF_CHECKPOINT="$REF_CKPT" \
  TESTER_EVAL_STEPS="$LONG_EVAL_STEPS" \
  TESTER_EVAL_THRESHOLD_Q="$THRESH_Q" \
  TESTER_EVAL_SPLITS="$LONG_EVAL_SPLITS" \
  TESTER_EVAL_RESUME_EXISTING=1 \
  TESTER_EVAL_SEMANTIC_FAULT_PROFILE="eval_holdout" \
  TESTER_EVAL_SEMANTIC_FAULT_EP_PROB="$SEMANTIC_EVAL_EP_PROB" \
  TESTER_EVAL_SEMANTIC_SUBTYPES="$SEMANTIC_HOLDOUT_SUBTYPES" \
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
    --script tester_eval \
    --configs crafter \
    --logdir "$outdir" \
    --seed 900 \
    --run.from_checkpoint "$ckpt" \
    --fault.enabled True \
    --fault.ref_ckpt "$REF_CKPT" \
    --fault.use_reward_error True \
    --fault.w_reward 0.0
}

run_final_analysis() {
  local outdir="$ROOT/analysis_long_resume"
  run_job "99_long_resume_analysis_summary" \
    "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/analyze_fault_results.py" \
      --roots "${PREV_ROOTS[@]}" "$CURRENT_ROOT" "$ROOT" \
      --outdir "$outdir" \
      --trace-analysis
  run_job "99_long_resume_analysis_context" \
    "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/context_conditioned_analysis.py" \
      --roots "${CONTEXT_ROOTS[@]}" "$CURRENT_ROOT" "$ROOT" \
      --outdir "$outdir"
}

record_status "fault_weekend_long_eval_resume" "START"
echo "===================================================="
echo "[Fault weekend long-eval resume]"
echo "root            : $ROOT"
echo "current root    : $CURRENT_ROOT"
echo "cases           : $CASES"
echo "steps           : $LONG_EVAL_STEPS"
echo "splits          : $LONG_EVAL_SPLITS"
echo "reference ckpt  : $REF_CKPT"
echo "===================================================="

for case_name in $(as_words "$CASES"); do
  ckpt="$(current_case_ckpt "$case_name")"
  if [[ -z "${ckpt:-}" || ! -d "$ckpt" ]]; then
    record_status "long_resume_${case_name}" "SKIPPED:no_ckpt"
    continue
  fi
  run_job "long_resume_${case_name}" run_tester_eval "long_eval_current_${case_name}" "$ckpt"
done

if [[ "$RUN_FINAL_ANALYSIS" == "1" ]]; then
  run_final_analysis
fi

record_status "fault_weekend_long_eval_resume" "FINISHED"
echo "Saved under: $ROOT"
