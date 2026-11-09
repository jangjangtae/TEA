#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$HOME/logdir/fault_3day_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"
CALIB_PY="$PROJECT_ROOT/dreamerv3/calibrate_fault_score.py"

REF_CKPT="${REF_CKPT:-/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487}"
CALIB_EPISODES="${CALIB_EPISODES:-1000}"
EVAL_STEPS="${EVAL_STEPS:-300000}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
REPLAY_SIZE="${REPLAY_SIZE:-3e5}"
THRESH_Q="${THRESH_Q:-0.99}"
FAULT_CLIP="${FAULT_CLIP:-1.0}"

mkdir -p "$ROOT"
STATUS_FILE="$ROOT/status.tsv"
STATS_FILE="$ROOT/calibration/clean_fault_stats.json"
CALIB_TRACE="$ROOT/calibration/clean_fault_trace.jsonl"

echo -e "time\tname\tstatus" > "$STATUS_FILE"

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

clean_env() {
  export CRAFTER_FAULT=0
  export CRAFTER_FAULT_SAMPLER=0
  export CRAFTER_SEMANTIC_FAULT_SAMPLER=0
  export CRAFTER_TESTER_REWARD=0
  export CRAFTER_USE_RND=0
  export CRAFTER_RND_UPDATE=0
  unset CRAFTER_FAULT_PROFILE || true
  unset CRAFTER_TRACE_PATH || true
  unset CRAFTER_OUTPUT_DIR || true
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
  find "$ckpt_root" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

run_calibration() {
  clean_env
  mkdir -p "$ROOT/calibration"
  export CRAFTER_OUTPUT_DIR="$ROOT/calibration/env"
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$CALIB_PY" \
    --configs crafter \
    --ref_ckpt "$REF_CKPT" \
    --episodes "$CALIB_EPISODES" \
    --out "$STATS_FILE" \
    --trace "$CALIB_TRACE" \
    --logdir "$ROOT/calibration" \
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

echo "===================================================="
echo "[Fault 3-day queue]"
echo "root          : $ROOT"
echo "reference ckpt: $REF_CKPT"
echo "calib episodes: $CALIB_EPISODES"
echo "eval steps    : $EVAL_STEPS"
echo "train steps   : $TRAIN_STEPS"
echo "===================================================="

run_job "01_calibration" run_calibration

if [[ -f "$STATS_FILE" ]]; then
  run_job "02_reference_tester_eval" run_tester_eval "reference" "$REF_CKPT"

  run_job "03_fault_adapt_beta01" run_fault_adapt "0.1"
  CKPT_BETA01="$(latest_ckpt_dir "$ROOT/fault_adapt_beta01")"
  if [[ -n "${CKPT_BETA01:-}" && -d "$CKPT_BETA01" ]]; then
    run_job "04_eval_fault_adapt_beta01" run_tester_eval "eval_fault_adapt_beta01" "$CKPT_BETA01"
  else
    record_status "04_eval_fault_adapt_beta01" "SKIPPED:no_ckpt"
  fi

  run_job "05_fault_adapt_beta02" run_fault_adapt "0.2"
  CKPT_BETA02="$(latest_ckpt_dir "$ROOT/fault_adapt_beta02")"
  if [[ -n "${CKPT_BETA02:-}" && -d "$CKPT_BETA02" ]]; then
    run_job "06_eval_fault_adapt_beta02" run_tester_eval "eval_fault_adapt_beta02" "$CKPT_BETA02"
  else
    record_status "06_eval_fault_adapt_beta02" "SKIPPED:no_ckpt"
  fi
else
  record_status "02_reference_tester_eval" "SKIPPED:no_calibration_stats"
  record_status "03_fault_adapt_beta01" "SKIPPED:no_calibration_stats"
  record_status "05_fault_adapt_beta02" "SKIPPED:no_calibration_stats"
fi

record_status "queue" "FINISHED"
echo "Saved under: $ROOT"
