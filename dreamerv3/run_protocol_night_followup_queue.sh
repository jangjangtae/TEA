#!/usr/bin/env bash
set -uo pipefail

# Night follow-up evaluations for the fault-score protocol.
# This waits for the current protocol eval to finish, then runs diagnostic
# eval-only queues that help separate context reachability, semantic transfer,
# and clean false-alarm stability.

CURRENT_ROOT="${CURRENT_ROOT:-/home/railab/logdir/fault_protocol_eval_20260609_145011}"
ROOT="${ROOT:-$HOME/logdir/fault_protocol_night_followup_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
QUEUE_SH="$PROJECT_ROOT/dreamerv3/run_semantic_holdout_eval_queue.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"

PROBE_EVAL_STEPS="${PROBE_EVAL_STEPS:-100000}"
CLEAN_EVAL_STEPS="${CLEAN_EVAL_STEPS:-200000}"
WAIT_FOR_CURRENT="${WAIT_FOR_CURRENT:-1}"
WAIT_STATUS_FILE="${WAIT_STATUS_FILE:-$CURRENT_ROOT/status.tsv}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"

RUN_SEMANTIC_PROB1="${RUN_SEMANTIC_PROB1:-1}"
RUN_SEMANTIC_TRAIN_PROBE="${RUN_SEMANTIC_TRAIN_PROBE:-1}"
RUN_SEMANTIC_TRIGGER_ONLY="${RUN_SEMANTIC_TRIGGER_ONLY:-0}"
RUN_CLEAN_LONG="${RUN_CLEAN_LONG:-1}"

SEMANTIC_HOLDOUT_SUBTYPES="${SEMANTIC_HOLDOUT_SUBTYPES:-tool_collect_desync_on_upgrade,craft_result_missing_on_retry,station_place_ghost_on_relocate,achievement_unlock_missing_after_valid_progress,station_usable_flag_broken_after_relocate,recipe_precondition_mischeck_on_retry,delayed_inventory_desync_after_station_use}"
SEMANTIC_TRAIN_SUBTYPES="${SEMANTIC_TRAIN_SUBTYPES:-collect_result_delayed_after_tool_upgrade,craft_output_delayed_on_retry,station_second_use_inconsistent_after_placement,progress_confirmation_requires_revisit,station_state_partial_reset_after_relocate,recipe_retry_requires_revisit}"

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

wait_for_current_queue() {
  if [[ "$WAIT_FOR_CURRENT" != "1" ]]; then
    return 0
  fi

  echo "Waiting for current protocol eval: $WAIT_STATUS_FILE"
  while true; do
    if [[ -f "$WAIT_STATUS_FILE" ]] && \
        grep -q $'semantic_holdout_eval_queue\tFINISHED' "$WAIT_STATUS_FILE"; then
      echo "Current protocol eval finished."
      return 0
    fi
    if [[ -f "$WAIT_STATUS_FILE" ]] && grep -q "FAILED:" "$WAIT_STATUS_FILE"; then
      echo "Current protocol eval has a FAILED status. Continuing follow-up anyway."
      return 0
    fi
    sleep "$CHECK_INTERVAL"
  done
}

run_queue() {
  local name="$1"
  shift
  local subroot="$ROOT/$name"
  local log_file="$ROOT/${name}.log"

  record_status "$name" "START"
  echo "====================================================" | tee -a "$log_file"
  echo "[$name] $(stamp)" | tee -a "$log_file"
  echo "Root: $subroot" | tee -a "$log_file"
  echo "Log : $log_file" | tee -a "$log_file"
  echo "====================================================" | tee -a "$log_file"

  env ROOT="$subroot" "$@" "$QUEUE_SH" 2>&1 | tee -a "$log_file"
  local code="${PIPESTATUS[0]}"
  if [[ "$code" -eq 0 ]]; then
    record_status "$name" "DONE"
  else
    record_status "$name" "FAILED:$code"
  fi
}

run_analysis() {
  local outdir="$ROOT/analysis"
  record_status "analysis" "START"
  "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/analyze_fault_results.py" \
    --roots \
      /home/railab/logdir/fault_3day_leave_20260528_170600 \
      /home/railab/logdir/fault_nextday_20260602_182008 \
      /home/railab/logdir/fault_followup3_20260604_175648 \
      /home/railab/logdir/fault_semantic_holdout_20260608_134437_systemd \
      "$CURRENT_ROOT" \
      "$ROOT" \
    --outdir "$outdir" \
    --trace-analysis
  local code="$?"
  if [[ "$code" -eq 0 ]]; then
    record_status "analysis" "DONE"
  else
    record_status "analysis" "FAILED:$code"
  fi
}

echo -e "time\tname\tstatus" > "$STATUS_FILE"

echo "===================================================="
echo "[Protocol night follow-up queue]"
echo "root              : $ROOT"
echo "current root      : $CURRENT_ROOT"
echo "probe eval steps  : $PROBE_EVAL_STEPS"
echo "clean eval steps  : $CLEAN_EVAL_STEPS"
echo "wait current      : $WAIT_FOR_CURRENT"
echo "trigger-only eval : $RUN_SEMANTIC_TRIGGER_ONLY"
echo "===================================================="

wait_for_current_queue

if [[ "$RUN_SEMANTIC_PROB1" == "1" ]]; then
  run_queue "01_semantic_holdout_prob1" \
    EVAL_STEPS="$PROBE_EVAL_STEPS" \
    EVAL_SPLITS="clean,semantic_holdout" \
    TESTER_EVAL_SEMANTIC_FAULT_EP_PROB="1.0" \
    SEMANTIC_MANIFEST_PROB="1.0" \
    SEMANTIC_SUBTYPES="$SEMANTIC_HOLDOUT_SUBTYPES" \
    RESUME_EXISTING="1"
fi

if [[ "$RUN_SEMANTIC_TRAIN_PROBE" == "1" ]]; then
  run_queue "02_semantic_train_probe_prob1" \
    EVAL_STEPS="$PROBE_EVAL_STEPS" \
    EVAL_SPLITS="clean,semantic_holdout" \
    TESTER_EVAL_SEMANTIC_FAULT_PROFILE="train" \
    TESTER_EVAL_SEMANTIC_FAULT_EP_PROB="1.0" \
    SEMANTIC_MANIFEST_PROB="1.0" \
    SEMANTIC_SUBTYPES="$SEMANTIC_TRAIN_SUBTYPES" \
    RESUME_EXISTING="1"
fi

if [[ "$RUN_SEMANTIC_TRIGGER_ONLY" == "1" ]]; then
  run_queue "03b_semantic_trigger_only_prob0" \
    EVAL_STEPS="$PROBE_EVAL_STEPS" \
    EVAL_SPLITS="clean,semantic_holdout" \
    TESTER_EVAL_SEMANTIC_FAULT_EP_PROB="1.0" \
    SEMANTIC_MANIFEST_PROB="0.0" \
    SEMANTIC_SUBTYPES="$SEMANTIC_HOLDOUT_SUBTYPES" \
    RESUME_EXISTING="1"
fi

if [[ "$RUN_CLEAN_LONG" == "1" ]]; then
  run_queue "03_clean_long_false_alarm" \
    EVAL_STEPS="$CLEAN_EVAL_STEPS" \
    EVAL_SPLITS="clean" \
    RESUME_EXISTING="1"
fi

run_analysis
record_status "protocol_night_followup_queue" "FINISHED"
echo "Saved under: $ROOT"
