#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# 기본 경로/설정
# =========================================================
ROOT="$HOME/dreamerv3"
PYTHON_BIN="python"
MAIN_PY="$ROOT/dreamerv3/main.py"

BASE_CKPT="/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487"

RUN_TAG="ablation_train_only_$(date +%Y%m%d_%H%M%S)"
LOGROOT="$HOME/logdir/$RUN_TAG"
mkdir -p "$LOGROOT"

TRAIN_STEPS=100000
REPLAY_SIZE="3e5"

# 공통 train 인자
COMMON_TRAIN_ARGS=(
  --script tester_train
  --configs crafter
  --run.from_checkpoint "$BASE_CKPT"
  --replay.size "$REPLAY_SIZE"
  --run.steps "$TRAIN_STEPS"
)

# =========================================================
# 공통 환경변수
#   - procedure reward / semantic bug 학습용
# =========================================================

# -------- tester reward 계열 --------
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

# -------- procedure reward 강도 --------
export CRAFTER_TESTER_REWARD=1
export CRAFTER_TESTER_ALPHA_TASK=1.0
export CRAFTER_TESTER_CTX_REWARD=0.02
export CRAFTER_TESTER_ANOM_REWARD=0.02
export CRAFTER_TESTER_REPRODUCE_REWARD=0.10
export CRAFTER_TESTER_COMPARE_REWARD=0.10
export CRAFTER_TESTER_CONFIRM_REWARD=0.12
export CRAFTER_TESTER_FOLLOWUP_REWARD=0.08
export CRAFTER_TESTER_REPEAT_PENALTY=0.01

# -------- legacy low-level fault 비활성화 --------
export CRAFTER_FAULT_SAMPLER=0
export CRAFTER_FAULT=0
unset CRAFTER_ACTION_SUBTYPES || true
unset CRAFTER_CONTEXT_SUBTYPES || true
unset CRAFTER_REWARD_SUBTYPES || true
unset CRAFTER_TERMINATION_SUBTYPES || true
unset CRAFTER_FAULT_PROFILE || true
unset CRAFTER_FAULT_FAMILIES || true
unset CRAFTER_TRACE_PATH || true

# -------- semantic high-level fault 활성화 --------
export CRAFTER_SEMANTIC_FAULT_SAMPLER=1
export CRAFTER_SEMANTIC_FAULT_PROFILE=eval_holdout
export CRAFTER_SEMANTIC_FAULT_EP_PROB=0.5
export CRAFTER_SEMANTIC_FAULT_VERBOSE=0
export CRAFTER_SEMANTIC_SUBTYPES=tool_collect_desync_on_upgrade,craft_result_missing_on_retry,station_place_ghost_on_relocate,achievement_unlock_missing_after_valid_progress,station_usable_flag_broken_after_relocate,recipe_precondition_mischeck_on_retry,delayed_inventory_desync_after_station_use

# =========================================================
# 헬퍼 함수
# =========================================================
run_case() {
  local name="$1"
  shift

  local train_dir="$LOGROOT/$name"
  mkdir -p "$train_dir"

  echo "===================================================="
  echo "[TRAIN] $name"
  echo "train dir : $train_dir"
  echo "steps     : $TRAIN_STEPS"
  echo "===================================================="

  export CRAFTER_OUTPUT_DIR="$train_dir"

  env -u LD_LIBRARY_PATH "$@" \
    "$PYTHON_BIN" "$MAIN_PY" \
    "${COMMON_TRAIN_ARGS[@]}" \
    --logdir "$train_dir"

  echo "[DONE] $name"
  echo
}

# =========================================================
# Ablation train only
# =========================================================

# ---------------------------------------------------------
# A1: Task only
#   - bug / coverage / detect 전부 끔
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# A2: Task + Bug
#   - global/local coverage 끄고, bug만 남김
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# A3: Task + Global Coverage
#   - anomaly bug term 없이 coverage만 유지
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# A4: Task + Bug + Coverage
#   - detect bonus만 제거
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# A5: Full
# ---------------------------------------------------------
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
echo "ALL TRAIN DONE"
echo "results saved under: $LOGROOT"
echo "===================================================="