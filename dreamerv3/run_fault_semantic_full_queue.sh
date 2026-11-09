#!/usr/bin/env bash
set -uo pipefail

# Comprehensive semantic follow-up queue.
# Questions:
#   1) Does fault-score reward help if it is gated to semantic contexts?
#   2) Does semantic-seen training transfer to semantic-unseen holdout bugs?
#   3) What does the procedural/tester reward upper bound look like?

ROOT="${ROOT:-$HOME/logdir/fault_semantic_full_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"

REF_CKPT="${REF_CKPT:-/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487}"
STATS_FILE="${STATS_FILE:-/home/railab/logdir/fault_3day_leave_20260528_170600/calibration/clean_fault_stats.json}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
EVAL_STEPS="${EVAL_STEPS:-100000}"
REPLAY_SIZE="${REPLAY_SIZE:-3e5}"
THRESH_Q="${THRESH_Q:-0.99}"
FAULT_CLIP="${FAULT_CLIP:-1.0}"
SEMANTIC_TRAIN_EP_PROB="${SEMANTIC_TRAIN_EP_PROB:-0.5}"
SEMANTIC_EVAL_EP_PROB="${SEMANTIC_EVAL_EP_PROB:-0.5}"

RUN_CONTEXT_ANALYSIS_BEFORE="${RUN_CONTEXT_ANALYSIS_BEFORE:-0}"
RUN_TASK_ONLY="${RUN_TASK_ONLY:-1}"
RUN_GATED_BETA005="${RUN_GATED_BETA005:-1}"
RUN_GATED_BETA01="${RUN_GATED_BETA01:-1}"
RUN_UNGATED_BETA005="${RUN_UNGATED_BETA005:-1}"
RUN_ORACLE_TESTER="${RUN_ORACLE_TESTER:-1}"
RUN_FINAL_ANALYSIS="${RUN_FINAL_ANALYSIS:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"

SEMANTIC_TRAIN_SUBTYPES="${SEMANTIC_TRAIN_SUBTYPES:-collect_result_delayed_after_tool_upgrade,craft_output_delayed_on_retry,station_second_use_inconsistent_after_placement,progress_confirmation_requires_revisit,station_state_partial_reset_after_relocate,recipe_retry_requires_revisit}"
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
  return 0
}

clear_fault_env() {
  export CRAFTER_FAULT_SAMPLER=0
  export CRAFTER_FAULT=0
  unset CRAFTER_ACTION_SUBTYPES || true
  unset CRAFTER_CONTEXT_SUBTYPES || true
  unset CRAFTER_REWARD_SUBTYPES || true
  unset CRAFTER_TERMINATION_SUBTYPES || true
  unset CRAFTER_FAULT_PROFILE || true
  unset CRAFTER_FAULT_FAMILIES || true
  unset CRAFTER_TRACE_PATH || true
}

set_semantic_train_env() {
  clear_fault_env
  export CRAFTER_SEMANTIC_FAULT_SAMPLER=1
  export CRAFTER_SEMANTIC_FAULT_PROFILE=train
  export CRAFTER_SEMANTIC_FAULT_EP_PROB="$SEMANTIC_TRAIN_EP_PROB"
  export CRAFTER_SEMANTIC_FAULT_VERBOSE=0
  export CRAFTER_SEMANTIC_SUBTYPES="$SEMANTIC_TRAIN_SUBTYPES"
  export CRAFTER_RECORD_GIFS=0
}

set_no_extra_rewards() {
  export CRAFTER_TESTER_REWARD=0
  export CRAFTER_USE_RND=0
  export CRAFTER_RND_UPDATE=0
}

set_oracle_tester_rewards() {
  export CRAFTER_TESTER_REWARD=1
  export CRAFTER_USE_RND=0
  export CRAFTER_RND_UPDATE=0

  export TESTER_BASELINE_SCORE="${TESTER_BASELINE_SCORE:-11.8}"
  export TESTER_GREEN_RATIO="${TESTER_GREEN_RATIO:-0.85}"
  export TESTER_YELLOW_RATIO="${TESTER_YELLOW_RATIO:-0.65}"
  export TESTER_REPEAT_BUDGET="${TESTER_REPEAT_BUDGET:-0.08}"
  export TESTER_INIT_LAMBDA_RECOVER="${TESTER_INIT_LAMBDA_RECOVER:-1.0}"
  export TESTER_INIT_LAMBDA_REPEAT="${TESTER_INIT_LAMBDA_REPEAT:-0.1}"
  export TESTER_MAX_LAMBDA_RECOVER="${TESTER_MAX_LAMBDA_RECOVER:-5.0}"
  export TESTER_MAX_LAMBDA_REPEAT="${TESTER_MAX_LAMBDA_REPEAT:-3.0}"
  export TESTER_ALPHA_TASK_BASE="${TESTER_ALPHA_TASK_BASE:-0.20}"
  export TESTER_ALPHA_COV_GLOBAL="${TESTER_ALPHA_COV_GLOBAL:-0.03}"
  export TESTER_ALPHA_DETECT="${TESTER_ALPHA_DETECT:-0.10}"
  export TESTER_INIT_W_BUG="${TESTER_INIT_W_BUG:-0.55}"
  export TESTER_MIN_W_BUG="${TESTER_MIN_W_BUG:-0.35}"
  export TESTER_MAX_W_BUG="${TESTER_MAX_W_BUG:-0.90}"
  export TESTER_INIT_BETA_COV="${TESTER_INIT_BETA_COV:-0.15}"
  export TESTER_MIN_BETA_COV="${TESTER_MIN_BETA_COV:-0.08}"
  export TESTER_MAX_BETA_COV="${TESTER_MAX_BETA_COV:-0.25}"
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

run_fault_train() {
  local outname="$1"
  local beta="$2"
  local log_only="$3"
  local reward_gate="$4"
  local outdir="$ROOT/$outname"
  mkdir -p "$outdir"

  set_semantic_train_env
  set_no_extra_rewards
  export CRAFTER_OUTPUT_DIR="$outdir"

  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
    --script train \
    --configs crafter \
    --logdir "$outdir" \
    --run.from_checkpoint "$REF_CKPT" \
    --run.steps "$TRAIN_STEPS" \
    --replay.size "$REPLAY_SIZE" \
    --fault.enabled True \
    --fault.ref_ckpt "$REF_CKPT" \
    --fault.norm_stats "$STATS_FILE" \
    --fault.log_only "$log_only" \
    --fault.beta "$beta" \
    --fault.clip "$FAULT_CLIP" \
    --fault.reward_gate "$reward_gate" \
    --fault.use_reward_error True \
    --fault.w_reward 0.0
}

run_oracle_tester_train() {
  local outname="$1"
  local outdir="$ROOT/$outname"
  mkdir -p "$outdir"

  set_semantic_train_env
  set_oracle_tester_rewards
  export CRAFTER_OUTPUT_DIR="$outdir"

  TESTER_REF_CHECKPOINT="$REF_CKPT" \
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
    --script tester_train \
    --configs crafter \
    --logdir "$outdir" \
    --run.from_checkpoint "$REF_CKPT" \
    --run.steps "$TRAIN_STEPS" \
    --replay.size "$REPLAY_SIZE"
}

run_tester_eval() {
  local outname="$1"
  local ckpt="$2"
  local semantic_profile="$3"
  local semantic_subtypes="$4"
  local splits="$5"
  local outdir="$ROOT/$outname"
  mkdir -p "$outdir"

  unset CRAFTER_OUTPUT_DIR || true
  unset CRAFTER_TRACE_PATH || true

  TESTER_EVAL_CHECKPOINT="$ckpt" \
  TESTER_REF_CHECKPOINT="$REF_CKPT" \
  TESTER_EVAL_STEPS="$EVAL_STEPS" \
  TESTER_EVAL_THRESHOLD_Q="$THRESH_Q" \
  TESTER_EVAL_SPLITS="$splits" \
  TESTER_EVAL_SEMANTIC_FAULT_PROFILE="$semantic_profile" \
  TESTER_EVAL_SEMANTIC_FAULT_EP_PROB="$SEMANTIC_EVAL_EP_PROB" \
  TESTER_EVAL_SEMANTIC_SUBTYPES="$semantic_subtypes" \
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

eval_case() {
  local case_name="$1"
  local ckpt="$2"
  if [[ -z "${ckpt:-}" || ! -d "$ckpt" ]]; then
    record_status "eval_${case_name}" "SKIPPED:no_ckpt"
    return 0
  fi
  run_job "eval_${case_name}_semantic_holdout" \
    run_tester_eval "eval_${case_name}_semantic_holdout" "$ckpt" \
    "eval_holdout" "$SEMANTIC_HOLDOUT_SUBTYPES" \
    "clean,seen,holdout,semantic_holdout"
  run_job "eval_${case_name}_semantic_train" \
    run_tester_eval "eval_${case_name}_semantic_train" "$ckpt" \
    "train" "$SEMANTIC_TRAIN_SUBTYPES" \
    "clean,semantic_holdout"
}

run_context_analysis() {
  local outdir="$ROOT/context_analysis"
  "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/context_conditioned_analysis.py" \
    --roots "${CONTEXT_ROOTS[@]}" "$ROOT" \
    --outdir "$outdir"
}

run_final_analysis() {
  local outdir="$ROOT/analysis"
  "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/analyze_fault_results.py" \
    --roots "${PREV_ROOTS[@]}" "$ROOT" \
    --outdir "$outdir" \
    --trace-analysis
  "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/context_conditioned_analysis.py" \
    --roots "${CONTEXT_ROOTS[@]}" "$ROOT" \
    --outdir "$outdir"
}

if [[ ! -f "$STATS_FILE" ]]; then
  echo "Missing calibration stats: $STATS_FILE" >&2
  exit 1
fi

echo -e "time\tname\tstatus" > "$STATUS_FILE"

echo "===================================================="
echo "[Fault semantic full queue]"
echo "root            : $ROOT"
echo "reference ckpt  : $REF_CKPT"
echo "stats           : $STATS_FILE"
echo "train steps     : $TRAIN_STEPS"
echo "eval steps      : $EVAL_STEPS"
echo "stop on fail    : $STOP_ON_FAIL"
echo "semantic train  : $SEMANTIC_TRAIN_SUBTYPES"
echo "semantic holdout: $SEMANTIC_HOLDOUT_SUBTYPES"
echo "===================================================="

if [[ "$RUN_CONTEXT_ANALYSIS_BEFORE" == "1" ]]; then
  run_job "00_context_conditioned_analysis_before" run_context_analysis
fi

if [[ "$RUN_TASK_ONLY" == "1" ]]; then
  run_job "01_semantic_task_only_fault_logging" \
    run_fault_train "semantic_task_only_fault_logging" "0.0" "True" "none"
  CKPT="$(latest_ckpt_dir "$ROOT/semantic_task_only_fault_logging")"
  eval_case "semantic_task_only_fault_logging" "$CKPT"
fi

if [[ "$RUN_GATED_BETA005" == "1" ]]; then
  run_job "02_semantic_fault_gated_beta005" \
    run_fault_train "semantic_fault_gated_beta005" "0.05" "False" "semantic_context"
  CKPT="$(latest_ckpt_dir "$ROOT/semantic_fault_gated_beta005")"
  eval_case "semantic_fault_gated_beta005" "$CKPT"
fi

if [[ "$RUN_GATED_BETA01" == "1" ]]; then
  run_job "03_semantic_fault_gated_beta01" \
    run_fault_train "semantic_fault_gated_beta01" "0.1" "False" "semantic_context"
  CKPT="$(latest_ckpt_dir "$ROOT/semantic_fault_gated_beta01")"
  eval_case "semantic_fault_gated_beta01" "$CKPT"
fi

if [[ "$RUN_UNGATED_BETA005" == "1" ]]; then
  run_job "04_semantic_fault_ungated_beta005" \
    run_fault_train "semantic_fault_ungated_beta005" "0.05" "False" "none"
  CKPT="$(latest_ckpt_dir "$ROOT/semantic_fault_ungated_beta005")"
  eval_case "semantic_fault_ungated_beta005" "$CKPT"
fi

if [[ "$RUN_ORACLE_TESTER" == "1" ]]; then
  run_job "05_semantic_oracle_tester_reward" \
    run_oracle_tester_train "semantic_oracle_tester_reward"
  CKPT="$(latest_ckpt_dir "$ROOT/semantic_oracle_tester_reward")"
  eval_case "semantic_oracle_tester_reward" "$CKPT"
fi

if [[ "$RUN_FINAL_ANALYSIS" == "1" ]]; then
  run_job "99_final_analysis" run_final_analysis
fi

record_status "fault_semantic_full_queue" "FINISHED"
echo "Saved under: $ROOT"
