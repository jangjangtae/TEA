#!/usr/bin/env bash
set -uo pipefail

# One-day queue for the paper-facing Crafter fault benchmark v2.
# Goal: get a quick but meaningful read by tomorrow without launching a
# multi-day sweep. Runs sanity eval, task-only adaptation, one threshold
# adaptation, and optional excess-delta adaptation.

BASE_ROOT="${BASE_ROOT:-/home/railab/logdir/fault_3day_leave_20260528_170600}"
REF_PROBE_ROOT="${REF_PROBE_ROOT:-/home/railab/logdir/fault_reference_protocol_probe_20260616_162218}"
ROOT="${ROOT:-$HOME/logdir/fault_v2_oneday_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"
ANALYZE_PY="$PROJECT_ROOT/dreamerv3/analyze_fault_results.py"

REF_CKPT="${REF_CKPT:-/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487}"
STATS_FILE="${STATS_FILE:-$BASE_ROOT/calibration/clean_fault_stats.json}"
TRAIN_STEPS="${TRAIN_STEPS:-200000}"
EVAL_STEPS="${EVAL_STEPS:-50000}"
SANITY_STEPS="${SANITY_STEPS:-10000}"
REPLAY_SIZE="${REPLAY_SIZE:-1e5}"
THRESH_Q="${THRESH_Q:-0.99}"
EVAL_SPLITS="${EVAL_SPLITS:-clean,seen,holdout}"

TRAIN_FAULT_PROFILE="${TRAIN_FAULT_PROFILE:-benchmark_v2_train}"
EVAL_SEEN_FAULT_PROFILE="${EVAL_SEEN_FAULT_PROFILE:-benchmark_v2_seen}"
EVAL_HOLDOUT_FAULT_PROFILE="${EVAL_HOLDOUT_FAULT_PROFILE:-benchmark_v2_holdout}"
FAULT_FREQ_TIER="${FAULT_FREQ_TIER:-benchmark}"
TRAIN_FAULT_EP_PROB="${TRAIN_FAULT_EP_PROB:-0.3}"

RUN_SANITY="${RUN_SANITY:-1}"
RUN_TASK_ONLY="${RUN_TASK_ONLY:-1}"
RUN_THRESHOLD="${RUN_THRESHOLD:-1}"
RUN_EXCESS_DELTA="${RUN_EXCESS_DELTA:-0}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
RESUME_EXISTING="${RESUME_EXISTING:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"

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

if [[ ! -f "$STATUS_FILE" || "$RESUME_EXISTING" != "1" ]]; then
  echo -e "time\tname\tstatus" > "$STATUS_FILE"
fi

record_status() {
  local name="$1"
  local status="$2"
  echo -e "$(stamp)\t$name\t$status" | tee -a "$STATUS_FILE"
}

job_done() {
  local name="$1"
  [[ "$RESUME_EXISTING" == "1" ]] && grep -Fq $'\t'"$name"$'\tDONE' "$STATUS_FILE"
}

run_job() {
  local name="$1"
  shift
  local log_file="$ROOT/${name}.log"

  if job_done "$name"; then
    echo "[$name] already DONE, skipping."
    return 0
  fi

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

train_fault_env() {
  export CRAFTER_FAULT=0
  export CRAFTER_FAULT_SAMPLER=1
  export CRAFTER_FAULT_PROFILE="$TRAIN_FAULT_PROFILE"
  export CRAFTER_FAULT_EP_PROB="$TRAIN_FAULT_EP_PROB"
  export CRAFTER_FAULT_FREQ_TIER="$FAULT_FREQ_TIER"
  export CRAFTER_SEMANTIC_FAULT_SAMPLER=0
  export CRAFTER_SEMANTIC_FAULT_EP_PROB=0.0
  export CRAFTER_SEMANTIC_FAULT_MANIFEST_PROB=0.0
  export CRAFTER_TESTER_REWARD=0
  export CRAFTER_USE_RND=0
  export CRAFTER_RND_UPDATE=0
  unset CRAFTER_TRACE_PATH || true
}

run_train_task_only() {
  local outdir="$ROOT/01_task_only_fault_logging"
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
    --fault.clip 1.0 \
    --fault.reward_gate none \
    --fault.use_reward_error True \
    --fault.w_reward 0.0
}

run_train_threshold() {
  local outdir="$ROOT/03_threshold_p95_beta01_action"
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
    --fault.beta 0.1 \
    --fault.reward_mode threshold \
    --fault.norm_mode p95 \
    --fault.reward_threshold 1.0 \
    --fault.reward_delta_threshold 0.5 \
    --fault.clip 1.0 \
    --fault.reward_gate nonzero_action \
    --fault.use_reward_error True \
    --fault.w_reward 0.0
}

run_train_excess_delta() {
  local outdir="$ROOT/05_excess_delta_p95_beta01_action"
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
    --fault.beta 0.1 \
    --fault.reward_mode excess_delta_threshold \
    --fault.norm_mode p95 \
    --fault.reward_threshold 1.0 \
    --fault.reward_delta_threshold 0.5 \
    --fault.clip 1.0 \
    --fault.reward_gate nonzero_action \
    --fault.use_reward_error True \
    --fault.w_reward 0.0
}

run_tester_eval() {
  local outname="$1"
  local ckpt="$2"
  local steps="${3:-$EVAL_STEPS}"
  local outdir="$ROOT/$outname"
  mkdir -p "$outdir"
  unset CRAFTER_OUTPUT_DIR || true
  unset CRAFTER_TRACE_PATH || true

  TESTER_EVAL_CHECKPOINT="$ckpt" \
  TESTER_REF_CHECKPOINT="$REF_CKPT" \
  TESTER_EVAL_STEPS="$steps" \
  TESTER_EVAL_THRESHOLD_Q="$THRESH_Q" \
  TESTER_EVAL_SPLITS="$EVAL_SPLITS" \
  TESTER_EVAL_FAULT_FREQ_TIER="$FAULT_FREQ_TIER" \
  TESTER_EVAL_CLEAN_FAULT_PROFILE="$TRAIN_FAULT_PROFILE" \
  TESTER_EVAL_SEEN_FAULT_PROFILE="$EVAL_SEEN_FAULT_PROFILE" \
  TESTER_EVAL_HOLDOUT_FAULT_PROFILE="$EVAL_HOLDOUT_FAULT_PROFILE" \
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

run_analysis() {
  local name="$1"
  local outdir="$ROOT/analysis"
  local roots=("$ROOT")
  if [[ -d "$REF_PROBE_ROOT" ]]; then
    roots=("$REF_PROBE_ROOT" "$ROOT")
  fi
  run_job "$name" \
    "$PYTHON_BIN" "$ANALYZE_PY" \
      --roots "${roots[@]}" \
      --outdir "$outdir" \
      --splits "$EVAL_SPLITS" \
      --trace-analysis \
      --top-fracs 0.01,0.05
}

run_eval_for_train() {
  local eval_job="$1"
  local eval_dir="$2"
  local train_dir="$3"
  local ckpt
  ckpt="$(latest_ckpt_dir "$ROOT/$train_dir")"
  if [[ -n "${ckpt:-}" && -d "$ckpt" ]]; then
    run_job "$eval_job" run_tester_eval "$eval_dir" "$ckpt"
  else
    record_status "$eval_job" "SKIPPED:no_ckpt"
  fi
}

if [[ ! -f "$STATS_FILE" ]]; then
  echo "Missing calibration stats: $STATS_FILE" >&2
  exit 1
fi

echo "===================================================="
echo "[Fault benchmark v2 one-day queue]"
echo "root              : $ROOT"
echo "reference ckpt    : $REF_CKPT"
echo "stats             : $STATS_FILE"
echo "train steps       : $TRAIN_STEPS"
echo "eval steps        : $EVAL_STEPS"
echo "sanity steps      : $SANITY_STEPS"
echo "train profile     : $TRAIN_FAULT_PROFILE"
echo "seen profile      : $EVAL_SEEN_FAULT_PROFILE"
echo "holdout profile   : $EVAL_HOLDOUT_FAULT_PROFILE"
echo "fault tier        : $FAULT_FREQ_TIER"
echo "run sanity        : $RUN_SANITY"
echo "run task-only     : $RUN_TASK_ONLY"
echo "run threshold     : $RUN_THRESHOLD"
echo "run excess-delta  : $RUN_EXCESS_DELTA"
echo "resume existing   : $RESUME_EXISTING"
echo "===================================================="

if [[ "$RUN_SANITY" == "1" ]]; then
  run_job "00_reference_v2_sanity_eval" \
    run_tester_eval "00_reference_v2_sanity_eval" "$REF_CKPT" "$SANITY_STEPS"
fi

if [[ "$RUN_TASK_ONLY" == "1" ]]; then
  run_job "01_task_only_fault_logging" run_train_task_only
  run_eval_for_train \
    "02_eval_task_only_fault_logging" \
    "02_eval_task_only_fault_logging" \
    "01_task_only_fault_logging"
  if [[ "$RUN_ANALYSIS" == "1" ]]; then
    run_analysis "90_analysis_after_task_only"
  fi
fi

if [[ "$RUN_THRESHOLD" == "1" ]]; then
  run_job "03_threshold_p95_beta01_action" run_train_threshold
  run_eval_for_train \
    "04_eval_threshold_p95_beta01_action" \
    "04_eval_threshold_p95_beta01_action" \
    "03_threshold_p95_beta01_action"
fi

if [[ "$RUN_EXCESS_DELTA" == "1" ]]; then
  run_job "05_excess_delta_p95_beta01_action" run_train_excess_delta
  run_eval_for_train \
    "06_eval_excess_delta_p95_beta01_action" \
    "06_eval_excess_delta_p95_beta01_action" \
    "05_excess_delta_p95_beta01_action"
fi

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  run_analysis "99_analysis"
fi

record_status "fault_v2_oneday_queue" "FINISHED"
echo "Saved under: $ROOT"
