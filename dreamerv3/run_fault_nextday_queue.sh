#!/usr/bin/env bash
set -uo pipefail

# Follow-up queue for unattended fault-score experiments.
# By default, this reuses the calibration stats from the current long run and
# waits for that run's resume queue to finish before launching new jobs.

BASE_ROOT="${BASE_ROOT:-/home/railab/logdir/fault_3day_leave_20260528_170600}"
ROOT="${ROOT:-$HOME/logdir/fault_nextday_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"

REF_CKPT="${REF_CKPT:-/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487}"
STATS_FILE="${STATS_FILE:-$BASE_ROOT/calibration/clean_fault_stats.json}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
EVAL_STEPS="${EVAL_STEPS:-300000}"
REPLAY_SIZE="${REPLAY_SIZE:-3e5}"
THRESH_Q="${THRESH_Q:-0.99}"
FAULT_CLIP="${FAULT_CLIP:-1.0}"

RUN_TASK_ONLY="${RUN_TASK_ONLY:-1}"
RUN_BETA05="${RUN_BETA05:-1}"
WAIT_FOR_CURRENT="${WAIT_FOR_CURRENT:-1}"
WAIT_STATUS_FILE="${WAIT_STATUS_FILE:-$BASE_ROOT/status.tsv}"

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
  fi
  return 0
}

wait_for_current_queue() {
  if [[ "$WAIT_FOR_CURRENT" != "1" ]]; then
    return 0
  fi
  echo "Waiting for current queue to finish: $WAIT_STATUS_FILE"
  while true; do
    if [[ -f "$WAIT_STATUS_FILE" ]] && grep -q $'resume_queue\tFINISHED' "$WAIT_STATUS_FILE"; then
      echo "Current queue finished."
      return 0
    fi
    if [[ -f "$WAIT_STATUS_FILE" ]] && grep -q $'queue\tFINISHED' "$WAIT_STATUS_FILE" \
        && grep -q $'06b_eval_fault_adapt_beta02\tDONE' "$WAIT_STATUS_FILE"; then
      echo "Current queue finished."
      return 0
    fi
    sleep 300
  done
}

train_fault_env() {
  export CRAFTER_FAULT=0
  export CRAFTER_FAULT_SAMPLER=1
  export CRAFTER_FAULT_PROFILE=train
  export CRAFTER_FAULT_EP_PROB="${CRAFTER_FAULT_EP_PROB:-0.3}"
  export CRAFTER_SEMANTIC_FAULT_SAMPLER=0
  export CRAFTER_TESTER_REWARD=0
  export CRAFTER_USE_RND=0
  export CRAFTER_RND_UPDATE=0
  unset CRAFTER_TRACE_PATH || true
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

run_task_only_adapt() {
  local outdir="$ROOT/task_only_fault_logging"
  mkdir -p "$outdir"
  train_fault_env
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
    --fault.log_only True \
    --fault.beta 0.0 \
    --fault.clip "$FAULT_CLIP" \
    --fault.use_reward_error True \
    --fault.w_reward 0.0
}

run_fault_adapt() {
  local beta="$1"
  local name="fault_adapt_beta${beta//./}"
  local outdir="$ROOT/$name"
  mkdir -p "$outdir"
  train_fault_env
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
    --fault.log_only False \
    --fault.beta "$beta" \
    --fault.clip "$FAULT_CLIP" \
    --fault.use_reward_error True \
    --fault.w_reward 0.0
}

run_tester_eval() {
  local name="$1"
  local ckpt="$2"
  local outdir="$ROOT/$name"
  mkdir -p "$outdir"
  unset CRAFTER_OUTPUT_DIR || true

  TESTER_EVAL_CHECKPOINT="$ckpt" \
  TESTER_REF_CHECKPOINT="$REF_CKPT" \
  TESTER_EVAL_STEPS="$EVAL_STEPS" \
  TESTER_EVAL_THRESHOLD_Q="$THRESH_Q" \
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

if [[ ! -f "$STATS_FILE" ]]; then
  echo "Missing calibration stats: $STATS_FILE" >&2
  exit 1
fi

echo -e "time\tname\tstatus" > "$STATUS_FILE"

echo "===================================================="
echo "[Fault next-day queue]"
echo "root          : $ROOT"
echo "base root     : $BASE_ROOT"
echo "reference ckpt: $REF_CKPT"
echo "stats         : $STATS_FILE"
echo "train steps   : $TRAIN_STEPS"
echo "eval steps    : $EVAL_STEPS"
echo "run task-only : $RUN_TASK_ONLY"
echo "run beta=0.5  : $RUN_BETA05"
echo "wait current  : $WAIT_FOR_CURRENT"
echo "===================================================="

wait_for_current_queue

if [[ "$RUN_TASK_ONLY" == "1" ]]; then
  run_job "01_task_only_fault_logging" run_task_only_adapt
  CKPT_TASK="$(latest_ckpt_dir "$ROOT/task_only_fault_logging")"
  if [[ -n "${CKPT_TASK:-}" && -d "$CKPT_TASK" ]]; then
    run_job "02_eval_task_only_fault_logging" run_tester_eval "eval_task_only_fault_logging" "$CKPT_TASK"
  else
    record_status "02_eval_task_only_fault_logging" "SKIPPED:no_ckpt"
  fi
fi

if [[ "$RUN_BETA05" == "1" ]]; then
  run_job "03_fault_adapt_beta05" run_fault_adapt "0.5"
  CKPT_BETA05="$(latest_ckpt_dir "$ROOT/fault_adapt_beta05")"
  if [[ -n "${CKPT_BETA05:-}" && -d "$CKPT_BETA05" ]]; then
    run_job "04_eval_fault_adapt_beta05" run_tester_eval "eval_fault_adapt_beta05" "$CKPT_BETA05"
  else
    record_status "04_eval_fault_adapt_beta05" "SKIPPED:no_ckpt"
  fi
fi

record_status "nextday_queue" "FINISHED"
echo "Saved under: $ROOT"
