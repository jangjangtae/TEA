#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# 기본 경로/설정
# =========================================================
ROOT="$HOME/logdir/ablation_train_only_20260401_191413"
PROJECT_ROOT="$HOME/dreamerv3"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"
PYTHON_BIN="python"

REF_CKPT="/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487"
EVAL_STEPS=100000
THRESH_Q=0.99

RUN_TAG="baseline_semantic_eval_100k_$(date +%Y%m%d_%H%M%S)"
OUTDIR="$ROOT/$RUN_TAG"
mkdir -p "$OUTDIR"

echo "===================================================="
echo "[BASELINE EVAL ONLY]"
echo "checkpoint : $REF_CKPT"
echo "logdir     : $OUTDIR"
echo "steps      : $EVAL_STEPS"
echo "===================================================="

# ---------------------------------------------------------
# legacy low-level fault 완전 비활성화
# ---------------------------------------------------------
export CRAFTER_FAULT_SAMPLER=0
export CRAFTER_FAULT=0
unset CRAFTER_ACTION_SUBTYPES || true
unset CRAFTER_CONTEXT_SUBTYPES || true
unset CRAFTER_REWARD_SUBTYPES || true
unset CRAFTER_TERMINATION_SUBTYPES || true
unset CRAFTER_FAULT_PROFILE || true
unset CRAFTER_FAULT_FAMILIES || true
unset CRAFTER_TRACE_PATH || true

# ---------------------------------------------------------
# semantic high-level fault 7개만 활성화
# ---------------------------------------------------------
export CRAFTER_SEMANTIC_FAULT_SAMPLER=1
export CRAFTER_SEMANTIC_FAULT_PROFILE=eval_holdout
export CRAFTER_SEMANTIC_FAULT_EP_PROB=0.5
export CRAFTER_SEMANTIC_FAULT_VERBOSE=0
export CRAFTER_SEMANTIC_SUBTYPES=tool_collect_desync_on_upgrade,craft_result_missing_on_retry,station_place_ghost_on_relocate,achievement_unlock_missing_after_valid_progress,station_usable_flag_broken_after_relocate,recipe_precondition_mischeck_on_retry,delayed_inventory_desync_after_station_use

# ---------------------------------------------------------
# env-side 로그를 전부 같은 폴더에 저장
# ---------------------------------------------------------
export CRAFTER_OUTPUT_DIR="$OUTDIR"

# tester_eval 설정
export TESTER_EVAL_CHECKPOINT="$REF_CKPT"
export TESTER_REF_CHECKPOINT="$REF_CKPT"
export TESTER_EVAL_STEPS="$EVAL_STEPS"
export TESTER_EVAL_THRESHOLD_Q="$THRESH_Q"

env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
  --script tester_eval \
  --configs crafter \
  --logdir "$OUTDIR" \
  --run.from_checkpoint "$REF_CKPT"

echo "===================================================="
echo "DONE"
echo "Saved under: $OUTDIR"
echo "===================================================="