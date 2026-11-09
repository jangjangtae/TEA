#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Full split train/eval for baseline + A1~A5
#   - Train: legacy / non-semantic faults only
#   - Eval : semantic high-level bug 7 only
#   - 50k train / 50k eval
# =========================================================

ROOT="$HOME/dreamerv3"
PYTHON_BIN="python"
MAIN_PY="$ROOT/dreamerv3/main.py"

BASE_CKPT="/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487"
REF_CKPT="$BASE_CKPT"

RUN_TAG="ablation_split_train_eval_50k_$(date +%Y%m%d_%H%M%S)"
LOGROOT="$HOME/logdir/$RUN_TAG"
mkdir -p "$LOGROOT"

TRAIN_STEPS=50000
EVAL_STEPS=50000
REPLAY_SIZE="3e5"

COMMON_TRAIN_ARGS=(
  --script tester_train
  --configs crafter
  --run.from_checkpoint "$BASE_CKPT"
  --replay.size "$REPLAY_SIZE"
  --run.steps "$TRAIN_STEPS"
)

COMMON_EVAL_ARGS=(
  --script tester_eval
  --configs crafter
)

# =========================================================
# 공통 env / reward 설정
# =========================================================

# -------- tester reward / gating 계열 --------
export TESTER_BASELINE_SCORE=11.8
export TESTER_GREEN_RATIO=0.85
export TESTER_YELLOW_RATIO=0.65
export TESTER_REPEAT_BUDGET=0.08

export TESTER_INIT_LAMBDA_RECOVER=1.0
export TESTER_INIT_LAMBDA_REPEAT=0.1
export TESTER_MAX_LAMBDA_RECOVER=5.0
export TESTER_MAX_LAMBDA_REPEAT=3.0

export TESTER_TASK_GATE_WARMUP=0.25
export TESTER_TASK_GATE_GREEN=0.20
export TESTER_TASK_GATE_YELLOW=0.25
export TESTER_TASK_GATE_RED=0.30

export TESTER_EXPLORE_GATE_WARMUP=0.85
export TESTER_EXPLORE_GATE_GREEN=0.75
export TESTER_EXPLORE_GATE_YELLOW=0.65
export TESTER_EXPLORE_GATE_RED=0.55

export TESTER_LAMBDA_RECOVER_UP_RED=0.12
export TESTER_LAMBDA_RECOVER_DECAY=0.997
export TESTER_LAMBDA_REP_LR=0.02

export TESTER_SUSPICION_EMA_ALPHA=0.10
export TESTER_SUSPICION_LOW=0.75
export TESTER_SUSPICION_HIGH=2.25
export TESTER_SUSPICION_ARM=0.55
export TESTER_LOCAL_WINDOW=8
export TESTER_DETECT_SUSPICION=0.75
export TESTER_DETECT_STREAK=3

export TESTER_TASK_Z_CLIP=5.0
export TESTER_BUG_Z_CLIP=5.0
export TESTER_REP_Z_CLIP=5.0
export TESTER_NORM_WARMUP=100
export TESTER_BUG_NORM_WARMUP=100

# -------- procedure reward v2 --------
export CRAFTER_TESTER_REWARD=1
export CRAFTER_TESTER_ALPHA_TASK=1.0
export CRAFTER_TESTER_CTX_REWARD=0.02
export CRAFTER_TESTER_ANOM_REWARD=0.02
export CRAFTER_TESTER_REPRODUCE_REWARD=0.10
export CRAFTER_TESTER_COMPARE_REWARD=0.10
export CRAFTER_TESTER_CONFIRM_REWARD=0.12
export CRAFTER_TESTER_FOLLOWUP_REWARD=0.08

export CRAFTER_TESTER_REPEAT_PENALTY=0.003
export CRAFTER_TESTER_SAME_ACTION_CAP=5
export CRAFTER_TESTER_REPEAT_MAX_STEP=0.03
export CRAFTER_TESTER_REPEAT_SUSPEND_ON_CONTEXT=1
export CRAFTER_TESTER_REPEAT_ONLY_ON_IDLE=1

# =========================================================
# fault mode helpers
# =========================================================
set_train_fault_mode() {
  # legacy / non-semantic fault만 사용
  export CRAFTER_FAULT_SAMPLER=1
  export CRAFTER_FAULT=1

  unset CRAFTER_TRACE_PATH || true

  # semantic high-level fault는 train에서 끔
  export CRAFTER_SEMANTIC_FAULT_SAMPLER=0
  export CRAFTER_SEMANTIC_FAULT_EP_PROB=0.0
  unset CRAFTER_SEMANTIC_FAULT_PROFILE || true
  unset CRAFTER_SEMANTIC_SUBTYPES || true
}

set_eval_fault_mode() {
  # legacy low-level fault 완전 비활성화
  export CRAFTER_FAULT_SAMPLER=0
  export CRAFTER_FAULT=0
  unset CRAFTER_ACTION_SUBTYPES || true
  unset CRAFTER_CONTEXT_SUBTYPES || true
  unset CRAFTER_REWARD_SUBTYPES || true
  unset CRAFTER_TERMINATION_SUBTYPES || true
  unset CRAFTER_FAULT_PROFILE || true
  unset CRAFTER_FAULT_FAMILIES || true
  unset CRAFTER_TRACE_PATH || true

  # semantic high-level fault 7개만 사용
  export CRAFTER_SEMANTIC_FAULT_SAMPLER=1
  export CRAFTER_SEMANTIC_FAULT_PROFILE=eval_holdout
  export CRAFTER_SEMANTIC_FAULT_EP_PROB=0.5
  export CRAFTER_SEMANTIC_FAULT_VERBOSE=0
  export CRAFTER_SEMANTIC_SUBTYPES=tool_collect_desync_on_upgrade,craft_result_missing_on_retry,station_place_ghost_on_relocate,achievement_unlock_missing_after_valid_progress,station_usable_flag_broken_after_relocate,recipe_precondition_mischeck_on_retry,delayed_inventory_desync_after_station_use
}

find_latest_ckpt_dir() {
  local train_dir="$1"
  find "$train_dir/ckpt" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

run_baseline_eval() {
  local outdir="$LOGROOT/baseline_eval"
  mkdir -p "$outdir"

  echo "===================================================="
  echo "[BASELINE EVAL]"
  echo "checkpoint : $BASE_CKPT"
  echo "eval dir   : $outdir"
  echo "steps      : $EVAL_STEPS"
  echo "===================================================="

  set_eval_fault_mode
  export CRAFTER_OUTPUT_DIR="$outdir"
  export TESTER_EVAL_CHECKPOINT="$BASE_CKPT"
  export TESTER_REF_CHECKPOINT="$REF_CKPT"
  export TESTER_EVAL_STEPS="$EVAL_STEPS"
  export TESTER_EVAL_THRESHOLD_Q="0.99"

  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
    "${COMMON_EVAL_ARGS[@]}" \
    --logdir "$outdir" \
    --run.from_checkpoint "$BASE_CKPT"

  echo "[DONE] baseline eval"
  echo
}

run_case() {
  local name="$1"
  shift

  local train_dir="$LOGROOT/$name"
  local eval_dir="$LOGROOT/${name}_eval"
  mkdir -p "$train_dir" "$eval_dir"

  echo "===================================================="
  echo "[TRAIN] $name"
  echo "train dir : $train_dir"
  echo "steps     : $TRAIN_STEPS"
  echo "===================================================="

  set_train_fault_mode
  export CRAFTER_OUTPUT_DIR="$train_dir"

  env -u LD_LIBRARY_PATH "$@" \
    "$PYTHON_BIN" "$MAIN_PY" \
    "${COMMON_TRAIN_ARGS[@]}" \
    --logdir "$train_dir"

  local ckpt_dir
  ckpt_dir="$(find_latest_ckpt_dir "$train_dir")"
  if [[ -z "${ckpt_dir:-}" ]]; then
    echo "[ERROR] checkpoint directory not found for $name"
    exit 1
  fi

  echo "----------------------------------------------------"
  echo "[EVAL] $name"
  echo "checkpoint : $ckpt_dir"
  echo "eval dir   : $eval_dir"
  echo "steps      : $EVAL_STEPS"
  echo "----------------------------------------------------"

  set_eval_fault_mode
  export CRAFTER_OUTPUT_DIR="$eval_dir"
  export TESTER_EVAL_CHECKPOINT="$ckpt_dir"
  export TESTER_REF_CHECKPOINT="$REF_CKPT"
  export TESTER_EVAL_STEPS="$EVAL_STEPS"
  export TESTER_EVAL_THRESHOLD_Q="0.99"

  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
    "${COMMON_EVAL_ARGS[@]}" \
    --logdir "$eval_dir" \
    --run.from_checkpoint "$ckpt_dir"

  echo "[DONE] $name"
  echo
}

# =========================================================
# baseline + A1~A5
# =========================================================
run_baseline_eval

# A1: Task only
run_case "a1_task_only" \
  TESTER_ALPHA_TASK_BASE=0.20 \
  TESTER_ALPHA_COV_GLOBAL=0.00 \
  TESTER_ALPHA_DETECT=0.00 \
  TESTER_INIT_W_BUG=0.00 \
  TESTER_MIN_W_BUG=0.00 \
  TESTER_MAX_W_BUG=0.00 \
  TESTER_INIT_BETA_COV=0.00 \
  TESTER_MIN_BETA_COV=0.00 \
  TESTER_MAX_BETA_COV=0.00

# A2: Task + Bug
run_case "a2_task_bug" \
  TESTER_ALPHA_TASK_BASE=0.20 \
  TESTER_ALPHA_COV_GLOBAL=0.00 \
  TESTER_ALPHA_DETECT=0.00 \
  TESTER_INIT_W_BUG=0.55 \
  TESTER_MIN_W_BUG=0.35 \
  TESTER_MAX_W_BUG=0.90 \
  TESTER_INIT_BETA_COV=0.00 \
  TESTER_MIN_BETA_COV=0.00 \
  TESTER_MAX_BETA_COV=0.00

# A3: Task + Global Coverage
run_case "a3_task_cov" \
  TESTER_ALPHA_TASK_BASE=0.20 \
  TESTER_ALPHA_COV_GLOBAL=0.03 \
  TESTER_ALPHA_DETECT=0.00 \
  TESTER_INIT_W_BUG=0.00 \
  TESTER_MIN_W_BUG=0.00 \
  TESTER_MAX_W_BUG=0.00 \
  TESTER_INIT_BETA_COV=0.15 \
  TESTER_MIN_BETA_COV=0.08 \
  TESTER_MAX_BETA_COV=0.25

# A4: Task + Bug + Coverage
run_case "a4_task_bug_cov" \
  TESTER_ALPHA_TASK_BASE=0.20 \
  TESTER_ALPHA_COV_GLOBAL=0.03 \
  TESTER_ALPHA_DETECT=0.00 \
  TESTER_INIT_W_BUG=0.55 \
  TESTER_MIN_W_BUG=0.35 \
  TESTER_MAX_W_BUG=0.90 \
  TESTER_INIT_BETA_COV=0.15 \
  TESTER_MIN_BETA_COV=0.08 \
  TESTER_MAX_BETA_COV=0.25

# A5: Full
run_case "a5_full_v17" \
  TESTER_ALPHA_TASK_BASE=0.20 \
  TESTER_ALPHA_COV_GLOBAL=0.03 \
  TESTER_ALPHA_DETECT=0.10 \
  TESTER_INIT_W_BUG=0.55 \
  TESTER_MIN_W_BUG=0.35 \
  TESTER_MAX_W_BUG=0.90 \
  TESTER_INIT_BETA_COV=0.15 \
  TESTER_MIN_BETA_COV=0.08 \
  TESTER_MAX_BETA_COV=0.25

echo "===================================================="
echo "ALL DONE"
echo "results saved under: $LOGROOT"
echo "===================================================="