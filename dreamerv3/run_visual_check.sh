#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Visual Sanity Check Script for Semantic Bugs
#   - Purpose: Quickly generate GIFs of semantic bug occurrences.
#   - Model: Baseline (Best progression capability to reach deep context).
#   - Steps: Reduced to 5,000 for fast verification.
# =========================================================

ROOT="$HOME/dreamerv3"
PYTHON_BIN="python"
MAIN_PY="$ROOT/dreamerv3/main.py"

# 게임을 가장 잘하는 Baseline 모델의 체크포인트 사용 (수정 필요 시 변경)
BASE_CKPT="/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487"

RUN_TAG="visual_sanity_check_$(date +%Y%m%d_%H%M%S)"
LOGROOT="$HOME/logdir/$RUN_TAG"
mkdir -p "$LOGROOT"

# 빠른 확인을 위해 스텝 수 대폭 축소
EVAL_STEPS=5000

COMMON_EVAL_ARGS=(
  --script tester_eval
  --configs crafter
)

# =========================================================
# [핵심] Visual Sanity Check 전용 환경변수
# =========================================================
export CRAFTER_RECORD_GIFS=1
export CRAFTER_SEMANTIC_FAULT_VERBOSE=1

# =========================================================
# helpers
# =========================================================
clear_legacy_fault_env() {
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

set_eval_semantic7_mode() {
  clear_legacy_fault_env

  # 평가용 7종 Semantic Bug 활성화
  export CRAFTER_SEMANTIC_FAULT_SAMPLER=1
  export CRAFTER_SEMANTIC_FAULT_PROFILE=eval_holdout
  export CRAFTER_SEMANTIC_FAULT_EP_PROB=0.5  # 버그 발생 확률을 50%로 높게 설정
  export CRAFTER_SEMANTIC_SUBTYPES=upgrade_branch_inconsistent_collect_behavior,craft_result_missing_on_retry,station_place_ghost_on_relocate,achievement_unlock_missing_after_reconfirm,station_usable_flag_broken_after_relocate,recipe_precondition_mischeck_on_retry,delayed_inventory_desync_after_station_use
}

# =========================================================
# run
# =========================================================
echo "===================================================="
echo "[VISUAL SANITY CHECK] Baseline Eval"
echo "checkpoint : $BASE_CKPT"
echo "eval dir   : $LOGROOT"
echo "steps      : $EVAL_STEPS"
echo "GIF Record : ON"
echo "===================================================="

set_eval_semantic7_mode
export CRAFTER_OUTPUT_DIR="$LOGROOT"
export TESTER_EVAL_CHECKPOINT="$BASE_CKPT"
export TESTER_EVAL_STEPS="$EVAL_STEPS"

env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
  "${COMMON_EVAL_ARGS[@]}" \
  --logdir "$LOGROOT" \
  --run.from_checkpoint "$BASE_CKPT"

echo "===================================================="
echo "SANITY CHECK DONE"
echo "Please check the generated GIFs in: $LOGROOT"
echo "===================================================="