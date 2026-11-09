#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# 기본 경로/설정
# =========================================================
ROOT="$HOME/dreamerv3"
PYTHON_BIN="python"
MAIN_PY="$ROOT/dreamerv3/main.py"

BASE_CKPT="/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487"
REF_CKPT="$BASE_CKPT"

RUN_TAG="ablation_$(date +%Y%m%d_%H%M%S)"
LOGROOT="$HOME/logdir/$RUN_TAG"
mkdir -p "$LOGROOT"

TRAIN_STEPS=300000
EVAL_STEPS=50000
REPLAY_SIZE="3e5"

# 공통 train 인자
COMMON_TRAIN_ARGS=(
  --script tester_train
  --configs crafter
  --run.from_checkpoint "$BASE_CKPT"
  --replay.size "$REPLAY_SIZE"
  --run.steps "$TRAIN_STEPS"
)

# 공통 eval 인자
COMMON_EVAL_ARGS=(
  --script tester_eval
  --configs crafter
)

# =========================================================
# 공통 환경변수
#   - 현재 v17 코드 기준, 너무 공격적이지 않게 기본값 설정
# =========================================================
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

# =========================================================
# 헬퍼 함수
# =========================================================
run_baseline_eval() {
  local eval_dir="$LOGROOT/baseline_eval"
  mkdir -p "$eval_dir"

  echo "===================================================="
  echo "[BASELINE EVAL] $eval_dir"
  echo "===================================================="

  TESTER_EVAL_CHECKPOINT="$BASE_CKPT" \
  TESTER_REF_CHECKPOINT="$BASE_CKPT" \
  TESTER_EVAL_STEPS="$EVAL_STEPS" \
  TESTER_EVAL_THRESHOLD_Q="0.99" \
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
    "${COMMON_EVAL_ARGS[@]}" \
    --logdir "$eval_dir" \
    --run.from_checkpoint "$BASE_CKPT"

  echo "[DONE] baseline eval"
}

find_latest_ckpt_dir() {
  local train_dir="$1"
  find "$train_dir/ckpt" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

run_case() {
  local name="$1"
  shift

  local train_dir="$LOGROOT/$name"
  local eval_dir="$LOGROOT/${name}_eval"

  mkdir -p "$train_dir" "$eval_dir"

  echo "===================================================="
  echo "[TRAIN] $name"
  echo "train dir: $train_dir"
  echo "===================================================="

  # 개별 케이스 env 적용
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

  echo "[EVAL] $name"
  echo "checkpoint: $ckpt_dir"

  TESTER_EVAL_CHECKPOINT="$ckpt_dir" \
  TESTER_REF_CHECKPOINT="$REF_CKPT" \
  TESTER_EVAL_STEPS="$EVAL_STEPS" \
  TESTER_EVAL_THRESHOLD_Q="0.99" \
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
    "${COMMON_EVAL_ARGS[@]}" \
    --logdir "$eval_dir" \
    --run.from_checkpoint "$ckpt_dir"

  echo "[DONE] $name"
}

# =========================================================
# Ablation 실행
#   - baseline은 먼저 평가
#   - 이후 5개 케이스 순차 실행
# =========================================================

run_baseline_eval

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
# A5: Full v17
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
echo "ALL DONE"
echo "results saved under: $LOGROOT"
echo "===================================================="