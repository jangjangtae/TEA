#!/usr/bin/env bash
set -uo pipefail

# Weekend follow-up queue for the fault-score protocol.
#
# It intentionally lives separate from run_fault_semantic_full_queue.sh so the
# currently running queue can finish untouched. By default this waits for that
# queue, runs longer evaluations for completed checkpoints, then runs a small
# set of additional ablations/repeats.

ROOT="${ROOT:-$HOME/logdir/fault_weekend_$(date +%Y%m%d_%H%M%S)}"
CURRENT_ROOT="${CURRENT_ROOT:-/home/railab/logdir/fault_semantic_full_20260610_142128}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"

REF_CKPT="${REF_CKPT:-/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487}"
STATS_FILE="${STATS_FILE:-/home/railab/logdir/fault_3day_leave_20260528_170600/calibration/clean_fault_stats.json}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
EVAL_STEPS="${EVAL_STEPS:-100000}"
LONG_EVAL_STEPS="${LONG_EVAL_STEPS:-200000}"
REPLAY_SIZE="${REPLAY_SIZE:-3e5}"
THRESH_Q="${THRESH_Q:-0.99}"
FAULT_CLIP="${FAULT_CLIP:-1.0}"
SEMANTIC_TRAIN_EP_PROB="${SEMANTIC_TRAIN_EP_PROB:-0.5}"
SEMANTIC_EVAL_EP_PROB="${SEMANTIC_EVAL_EP_PROB:-0.5}"

WAIT_FOR_CURRENT="${WAIT_FOR_CURRENT:-1}"
CURRENT_STATUS_FILE="${CURRENT_STATUS_FILE:-$CURRENT_ROOT/status.tsv}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
CONTINUE_ON_CURRENT_FAIL="${CONTINUE_ON_CURRENT_FAIL:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
USE_ENV_SEED="${USE_ENV_SEED:-0}"
RESET_STATUS="${RESET_STATUS:-1}"

RUN_LONG_EVAL_CURRENT="${RUN_LONG_EVAL_CURRENT:-1}"
RUN_GATE_ABLATIONS="${RUN_GATE_ABLATIONS:-1}"
RUN_SEED_REPEATS="${RUN_SEED_REPEATS:-1}"
RUN_FINAL_ANALYSIS="${RUN_FINAL_ANALYSIS:-1}"

# Comma or space separated values are accepted.
SEEDS="${SEEDS:-1}"
SEED_REPEAT_CASES="${SEED_REPEAT_CASES:-task_only,gated_beta005}"
GATE_ABLATIONS="${GATE_ABLATIONS:-semantic_or_reward:0.05}"
LONG_EVAL_CASES="${LONG_EVAL_CASES:-task_only,gated_beta005,gated_beta01,ungated_beta005,oracle_tester}"
LONG_EVAL_SPLITS="${LONG_EVAL_SPLITS:-clean,seen,holdout,semantic_holdout}"

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

as_words() {
  echo "${1//,/ }"
}

seed_flags() {
  local seed="$1"
  printf '%s\n' --seed "$seed"
  if [[ "$USE_ENV_SEED" == "1" ]]; then
    printf '%s\n' --env.crafter.use_seed True
  fi
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

wait_for_current_queue() {
  if [[ "$WAIT_FOR_CURRENT" != "1" ]]; then
    record_status "wait_for_current_queue" "SKIPPED"
    return 0
  fi

  record_status "wait_for_current_queue" "START"
  echo "Waiting for current queue status: $CURRENT_STATUS_FILE"
  while true; do
    if [[ -f "$CURRENT_STATUS_FILE" ]] && \
        grep -q $'fault_semantic_full_queue\tFINISHED' "$CURRENT_STATUS_FILE"; then
      record_status "wait_for_current_queue" "DONE"
      return 0
    fi
    if [[ -f "$CURRENT_STATUS_FILE" ]] && grep -q "FAILED:" "$CURRENT_STATUS_FILE"; then
      if [[ "$CONTINUE_ON_CURRENT_FAIL" == "1" ]]; then
        record_status "wait_for_current_queue" "DONE:current_failed_continue"
        return 0
      fi
      record_status "wait_for_current_queue" "FAILED:current_queue_failed"
      exit 1
    fi
    sleep "$CHECK_INTERVAL"
  done
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
    reference)
      echo "$REF_CKPT"
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
    ungated_beta005)
      latest_ckpt_dir "$CURRENT_ROOT/semantic_fault_ungated_beta005"
      ;;
    oracle_tester)
      latest_ckpt_dir "$CURRENT_ROOT/semantic_oracle_tester_reward"
      ;;
    *)
      echo ""
      ;;
  esac
}

run_fault_train() {
  local outname="$1"
  local beta="$2"
  local log_only="$3"
  local reward_gate="$4"
  local seed="$5"
  local outdir="$ROOT/$outname"
  mkdir -p "$outdir"

  set_semantic_train_env
  set_no_extra_rewards
  export CRAFTER_OUTPUT_DIR="$outdir"

  mapfile -t extra_seed_flags < <(seed_flags "$seed")

  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
    --script train \
    --configs crafter \
    --logdir "$outdir" \
    --seed "$seed" \
    "${extra_seed_flags[@]:2}" \
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

run_tester_eval() {
  local outname="$1"
  local ckpt="$2"
  local semantic_profile="$3"
  local semantic_subtypes="$4"
  local splits="$5"
  local eval_steps="$6"
  local seed="$7"
  local outdir="$ROOT/$outname"
  mkdir -p "$outdir"

  unset CRAFTER_OUTPUT_DIR || true
  unset CRAFTER_TRACE_PATH || true
  mapfile -t extra_seed_flags < <(seed_flags "$seed")

  TESTER_EVAL_CHECKPOINT="$ckpt" \
  TESTER_REF_CHECKPOINT="$REF_CKPT" \
  TESTER_EVAL_STEPS="$eval_steps" \
  TESTER_EVAL_THRESHOLD_Q="$THRESH_Q" \
  TESTER_EVAL_SPLITS="$splits" \
  TESTER_EVAL_RESUME_EXISTING=1 \
  TESTER_EVAL_SEMANTIC_FAULT_PROFILE="$semantic_profile" \
  TESTER_EVAL_SEMANTIC_FAULT_EP_PROB="$SEMANTIC_EVAL_EP_PROB" \
  TESTER_EVAL_SEMANTIC_SUBTYPES="$semantic_subtypes" \
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
    --script tester_eval \
    --configs crafter \
    --logdir "$outdir" \
    --seed "$seed" \
    "${extra_seed_flags[@]:2}" \
    --run.from_checkpoint "$ckpt" \
    --fault.enabled True \
    --fault.ref_ckpt "$REF_CKPT" \
    --fault.use_reward_error True \
    --fault.w_reward 0.0
}

eval_case() {
  local case_name="$1"
  local ckpt="$2"
  local eval_steps="${3:-$EVAL_STEPS}"
  local seed="${4:-0}"
  if [[ -z "${ckpt:-}" || ! -d "$ckpt" ]]; then
    record_status "eval_${case_name}" "SKIPPED:no_ckpt"
    return 0
  fi
  run_job "eval_${case_name}_semantic_holdout" \
    run_tester_eval "eval_${case_name}_semantic_holdout" "$ckpt" \
    "eval_holdout" "$SEMANTIC_HOLDOUT_SUBTYPES" \
    "clean,seen,holdout,semantic_holdout" "$eval_steps" "$seed"
  run_job "eval_${case_name}_semantic_train" \
    run_tester_eval "eval_${case_name}_semantic_train" "$ckpt" \
    "train" "$SEMANTIC_TRAIN_SUBTYPES" \
    "clean,semantic_holdout" "$eval_steps" "$seed"
}

seed_case_params() {
  local case_name="$1"
  case "$case_name" in
    task_only)
      echo "semantic_task_only_fault_logging 0.0 True none"
      ;;
    gated_beta005)
      echo "semantic_fault_gated_beta005 0.05 False semantic_context"
      ;;
    gated_beta01)
      echo "semantic_fault_gated_beta01 0.1 False semantic_context"
      ;;
    ungated_beta005)
      echo "semantic_fault_ungated_beta005 0.05 False none"
      ;;
    *)
      echo ""
      ;;
  esac
}

run_long_eval_current() {
  for case_name in $(as_words "$LONG_EVAL_CASES"); do
    local ckpt
    ckpt="$(current_case_ckpt "$case_name")"
    if [[ -z "${ckpt:-}" || ! -d "$ckpt" ]]; then
      record_status "long_eval_current_${case_name}" "SKIPPED:no_ckpt"
      continue
    fi
    run_job "long_eval_current_${case_name}" \
      run_tester_eval "long_eval_current_${case_name}" "$ckpt" \
      "eval_holdout" "$SEMANTIC_HOLDOUT_SUBTYPES" \
      "$LONG_EVAL_SPLITS" "$LONG_EVAL_STEPS" "900"
  done
}

run_gate_ablations() {
  for spec in $(as_words "$GATE_ABLATIONS"); do
    local gate="${spec%%:*}"
    local beta="${spec##*:}"
    if [[ "$gate" == "$beta" ]]; then
      beta="0.05"
    fi
    local beta_label="${beta//./}"
    local gate_label="${gate//[^A-Za-z0-9]/_}"
    local outname="semantic_fault_gate_${gate_label}_beta${beta_label}"

    run_job "gate_ablation_${gate_label}_beta${beta_label}" \
      run_fault_train "$outname" "$beta" "False" "$gate" "10"
    local ckpt
    ckpt="$(latest_ckpt_dir "$ROOT/$outname")"
    eval_case "$outname" "$ckpt" "$EVAL_STEPS" "910"
  done
}

run_seed_repeats() {
  for seed in $(as_words "$SEEDS"); do
    for case_name in $(as_words "$SEED_REPEAT_CASES"); do
      local params
      params="$(seed_case_params "$case_name")"
      if [[ -z "$params" ]]; then
        record_status "seed${seed}_${case_name}" "SKIPPED:unknown_case"
        continue
      fi

      read -r base_outname beta log_only gate <<< "$params"
      local outname="seed${seed}_${base_outname}"
      run_job "seed${seed}_${case_name}" \
        run_fault_train "$outname" "$beta" "$log_only" "$gate" "$seed"
      local ckpt
      ckpt="$(latest_ckpt_dir "$ROOT/$outname")"
      eval_case "$outname" "$ckpt" "$EVAL_STEPS" "$((seed + 1000))"
    done
  done
}

run_final_analysis() {
  local outdir="$ROOT/analysis"
  run_job "99_final_analysis_summary" \
    "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/analyze_fault_results.py" \
      --roots "${PREV_ROOTS[@]}" "$CURRENT_ROOT" "$ROOT" \
      --outdir "$outdir" \
      --trace-analysis
  run_job "99_final_analysis_context" \
    "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/context_conditioned_analysis.py" \
      --roots "${CONTEXT_ROOTS[@]}" "$CURRENT_ROOT" "$ROOT" \
      --outdir "$outdir"
}

if [[ ! -f "$STATS_FILE" ]]; then
  echo "Missing calibration stats: $STATS_FILE" >&2
  exit 1
fi

if [[ "$RESET_STATUS" == "1" || ! -f "$STATUS_FILE" ]]; then
  echo -e "time\tname\tstatus" > "$STATUS_FILE"
else
  record_status "fault_weekend_queue_resume" "APPEND_STATUS"
fi

echo "===================================================="
echo "[Fault weekend queue]"
echo "root                 : $ROOT"
echo "current root         : $CURRENT_ROOT"
echo "wait current         : $WAIT_FOR_CURRENT"
echo "train steps          : $TRAIN_STEPS"
echo "eval steps           : $EVAL_STEPS"
echo "long eval steps      : $LONG_EVAL_STEPS"
echo "seeds                : $SEEDS"
echo "seed repeat cases    : $SEED_REPEAT_CASES"
echo "gate ablations       : $GATE_ABLATIONS"
echo "long eval cases      : $LONG_EVAL_CASES"
echo "long eval splits     : $LONG_EVAL_SPLITS"
echo "use env seed         : $USE_ENV_SEED"
echo "reset status         : $RESET_STATUS"
echo "stop on fail         : $STOP_ON_FAIL"
echo "===================================================="

wait_for_current_queue

if [[ "$RUN_LONG_EVAL_CURRENT" == "1" ]]; then
  run_job "00_long_eval_current_block" run_long_eval_current
fi

if [[ "$RUN_GATE_ABLATIONS" == "1" ]]; then
  run_job "01_gate_ablation_block" run_gate_ablations
fi

if [[ "$RUN_SEED_REPEATS" == "1" ]]; then
  run_job "02_seed_repeat_block" run_seed_repeats
fi

if [[ "$RUN_FINAL_ANALYSIS" == "1" ]]; then
  run_final_analysis
fi

record_status "fault_weekend_queue" "FINISHED"
echo "Saved under: $ROOT"
