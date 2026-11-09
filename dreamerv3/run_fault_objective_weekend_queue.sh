#!/usr/bin/env bash
set -uo pipefail

# Weekend queue for fault-reward objective ablations.
# Reuses the previous task-only/dense long run as a comparison root and runs
# threshold-style objectives that should behave less like dense novelty reward.

BASE_ROOT="${BASE_ROOT:-/home/railab/logdir/fault_3day_leave_20260528_170600}"
COMPARISON_ROOT="${COMPARISON_ROOT:-/home/railab/logdir/fault_profile_long_20260618_134300}"
REF_PROBE_ROOT="${REF_PROBE_ROOT:-/home/railab/logdir/fault_reference_protocol_probe_20260616_162218}"
ROOT="${ROOT:-$HOME/logdir/fault_objective_weekend_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"
ANALYZE_PY="$PROJECT_ROOT/dreamerv3/analyze_fault_results.py"
PLOT_PY="$PROJECT_ROOT/dreamerv3/plot_fault_objective_results.py"

REF_CKPT="${REF_CKPT:-/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487}"
STATS_FILE="${STATS_FILE:-$BASE_ROOT/calibration/clean_fault_stats.json}"
TRAIN_STEPS="${TRAIN_STEPS:-200000}"
EVAL_STEPS="${EVAL_STEPS:-75000}"
REPLAY_SIZE="${REPLAY_SIZE:-1e5}"
THRESH_Q="${THRESH_Q:-0.99}"
EVAL_SPLITS="${EVAL_SPLITS:-clean,seen,holdout}"
TRAIN_FAULT_EP_PROB="${TRAIN_FAULT_EP_PROB:-0.3}"
TRAIN_FAULT_PROFILE="${TRAIN_FAULT_PROFILE:-benchmark_train}"
EVAL_SEEN_FAULT_PROFILE="${EVAL_SEEN_FAULT_PROFILE:-benchmark_seen}"
EVAL_HOLDOUT_FAULT_PROFILE="${EVAL_HOLDOUT_FAULT_PROFILE:-benchmark_holdout}"
FAULT_FREQ_TIER="${FAULT_FREQ_TIER:-benchmark}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"

RUN_THRESH_P95="${RUN_THRESH_P95:-1}"
RUN_THRESH_P99="${RUN_THRESH_P99:-1}"
RUN_EXCESS_P95="${RUN_EXCESS_P95:-1}"
RUN_DELTA_P95="${RUN_DELTA_P95:-1}"
RUN_EXCESS_DELTA_P95="${RUN_EXCESS_DELTA_P95:-1}"

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

run_train() {
  local outname="$1"
  local beta="$2"
  local reward_mode="$3"
  local norm_mode="$4"
  local reward_threshold="$5"
  local clip="$6"
  local reward_gate="$7"
  local delta_threshold="${8:-0.5}"
  local outdir="$ROOT/$outname"
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
    --fault.reward_mode "$reward_mode" \
    --fault.norm_mode "$norm_mode" \
    --fault.reward_threshold "$reward_threshold" \
    --fault.reward_delta_threshold "$delta_threshold" \
    --fault.clip "$clip" \
    --fault.reward_gate "$reward_gate" \
    --fault.use_reward_error True \
    --fault.w_reward 0.0
}

run_tester_eval() {
  local outname="$1"
  local ckpt="$2"
  local outdir="$ROOT/$outname"
  mkdir -p "$outdir"
  unset CRAFTER_OUTPUT_DIR || true

  TESTER_EVAL_CHECKPOINT="$ckpt" \
  TESTER_REF_CHECKPOINT="$REF_CKPT" \
  TESTER_EVAL_STEPS="$EVAL_STEPS" \
  TESTER_EVAL_THRESHOLD_Q="$THRESH_Q" \
  TESTER_EVAL_SPLITS="$EVAL_SPLITS" \
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

run_case() {
  local train_name="$1"
  local eval_name="$2"
  local beta="$3"
  local reward_mode="$4"
  local norm_mode="$5"
  local reward_threshold="$6"
  local clip="$7"
  local reward_gate="$8"
  local delta_threshold="${9:-0.5}"

  run_job "$train_name" \
    run_train "$train_name" "$beta" "$reward_mode" "$norm_mode" \
      "$reward_threshold" "$clip" "$reward_gate" "$delta_threshold"
  local ckpt
  ckpt="$(latest_ckpt_dir "$ROOT/$train_name")"
  if [[ -n "${ckpt:-}" && -d "$ckpt" ]]; then
    run_job "$eval_name" run_tester_eval "$eval_name" "$ckpt"
  else
    record_status "$eval_name" "SKIPPED:no_ckpt"
  fi
}

run_analysis() {
  local outdir="$ROOT/analysis"
  local roots=()
  if [[ -d "$REF_PROBE_ROOT" ]]; then
    roots+=("$REF_PROBE_ROOT")
  fi
  if [[ -d "$COMPARISON_ROOT" ]]; then
    roots+=("$COMPARISON_ROOT")
  fi
  roots+=("$ROOT")

  run_job "99_analysis" \
    "$PYTHON_BIN" "$ANALYZE_PY" \
      --roots "${roots[@]}" \
      --outdir "$outdir" \
      --splits "$EVAL_SPLITS" \
      --trace-analysis \
      --top-fracs 0.01,0.05

  if [[ -f "$PLOT_PY" ]]; then
    run_job "99_objective_plots" \
      "$PYTHON_BIN" "$PLOT_PY" \
        --analysis-dir "$outdir"
  fi
}

if [[ ! -f "$STATS_FILE" ]]; then
  echo "Missing calibration stats: $STATS_FILE" >&2
  exit 1
fi

echo -e "time\tname\tstatus" > "$STATUS_FILE"

echo "===================================================="
echo "[Fault objective weekend queue]"
echo "root              : $ROOT"
echo "comparison root   : $COMPARISON_ROOT"
echo "reference ckpt    : $REF_CKPT"
echo "reference probe   : $REF_PROBE_ROOT"
echo "stats             : $STATS_FILE"
echo "train steps       : $TRAIN_STEPS"
echo "eval steps        : $EVAL_STEPS"
echo "eval splits       : $EVAL_SPLITS"
echo "replay size       : $REPLAY_SIZE"
echo "fault freq tier   : $FAULT_FREQ_TIER"
echo "train profile     : $TRAIN_FAULT_PROFILE"
echo "seen profile      : $EVAL_SEEN_FAULT_PROFILE"
echo "holdout profile   : $EVAL_HOLDOUT_FAULT_PROFILE"
echo "run threshold p95 : $RUN_THRESH_P95"
echo "run threshold p99 : $RUN_THRESH_P99"
echo "run excess p95    : $RUN_EXCESS_P95"
echo "run delta p95     : $RUN_DELTA_P95"
echo "run excess delta  : $RUN_EXCESS_DELTA_P95"
echo "===================================================="

if [[ "$RUN_THRESH_P95" == "1" ]]; then
  run_case \
    "01_threshold_p95_beta01_action" \
    "02_eval_threshold_p95_beta01_action" \
    "0.1" "threshold" "p95" "1.0" "1.0" "nonzero_action" "0.5"
fi

if [[ "$RUN_THRESH_P99" == "1" ]]; then
  run_case \
    "03_threshold_p99_beta02_action" \
    "04_eval_threshold_p99_beta02_action" \
    "0.2" "threshold" "p99" "1.0" "1.0" "nonzero_action" "0.5"
fi

if [[ "$RUN_EXCESS_P95" == "1" ]]; then
  run_case \
    "05_excess_p95_beta01_action" \
    "06_eval_excess_p95_beta01_action" \
    "0.1" "excess_threshold" "p95" "1.0" "2.0" "nonzero_action" "0.5"
fi

if [[ "$RUN_DELTA_P95" == "1" ]]; then
  run_case \
    "07_delta_p95_beta01_action" \
    "08_eval_delta_p95_beta01_action" \
    "0.1" "delta_threshold" "p95" "1.0" "1.0" "nonzero_action" "0.5"
fi

if [[ "$RUN_EXCESS_DELTA_P95" == "1" ]]; then
  run_case \
    "09_excess_delta_p95_beta01_action" \
    "10_eval_excess_delta_p95_beta01_action" \
    "0.1" "excess_delta_threshold" "p95" "1.0" "2.0" "nonzero_action" "0.5"
fi

run_analysis
record_status "fault_objective_weekend_queue" "FINISHED"
echo "Saved under: $ROOT"
