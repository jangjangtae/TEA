#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/dreamerv3"
PYTHON_BIN="python"
MAIN_PY="$ROOT/dreamerv3/main.py"

BASE_CKPT="/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487"
RND_CKPT="/home/railab/logdir/baseline_clean_full_rp3e5/ckpt/20260415T200042F321550"

RUN_TAG="bug_adaptation_100k_$(date +%Y%m%d_%H%M%S)"
LOGROOT="$HOME/logdir/$RUN_TAG"
mkdir -p "$LOGROOT"

TRAIN_STEPS=100000
EVAL_STEPS=100000
REPLAY_SIZE="3e5"

COMMON_TRAIN_ARGS=(
  --script tester_train
  --configs crafter
  --replay.size "$REPLAY_SIZE"
  --run.steps "$TRAIN_STEPS"
)

COMMON_EVAL_ARGS=(
  --script tester_eval
  --configs crafter
)

# =========================================================
# Common environment settings
# =========================================================
export CRAFTER_RECORD_GIFS=0
export CRAFTER_SEMANTIC_FAULT_VERBOSE=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

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

set_train_split_mode() {
  clear_legacy_fault_env
  export CRAFTER_SEMANTIC_FAULT_SAMPLER=1
  export CRAFTER_SEMANTIC_FAULT_PROFILE=train
  export CRAFTER_SEMANTIC_FAULT_EP_PROB=0.5
  export CRAFTER_SEMANTIC_SUBTYPES=collect_result_delayed_after_tool_upgrade,craft_output_delayed_on_retry,station_second_use_inconsistent_after_placement,progress_confirmation_requires_revisit,station_state_partial_reset_after_relocate,recipe_retry_requires_revisit
}

set_eval_semantic7_mode() {
  clear_legacy_fault_env
  export CRAFTER_SEMANTIC_FAULT_SAMPLER=1
  export CRAFTER_SEMANTIC_FAULT_PROFILE=eval_holdout
  export CRAFTER_SEMANTIC_FAULT_EP_PROB=0.5
  export CRAFTER_SEMANTIC_SUBTYPES=upgrade_branch_inconsistent_collect_behavior,craft_result_missing_on_retry,station_place_ghost_on_relocate,achievement_unlock_missing_after_reconfirm,station_usable_flag_broken_after_relocate,recipe_precondition_mischeck_on_retry,delayed_inventory_desync_after_station_use
}

find_latest_ckpt_dir() {
  local train_dir="$1"
  find "$train_dir/ckpt" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

run_case() {
  local name="$1"
  local from_ckpt="$2"
  shift 2

  local train_dir="$LOGROOT/$name"
  local eval_dir="$LOGROOT/${name}_eval"
  mkdir -p "$train_dir" "$eval_dir"

  echo "===================================================="
  echo "[TRAIN] $name"
  echo "from ckpt : $from_ckpt"
  echo "train dir : $train_dir"
  echo "===================================================="

  set_train_split_mode
  export CRAFTER_OUTPUT_DIR="$train_dir"

  env -u LD_LIBRARY_PATH "$@" \
    "$PYTHON_BIN" "$MAIN_PY" \
    "${COMMON_TRAIN_ARGS[@]}" \
    --run.from_checkpoint "$from_ckpt" \
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
  echo "----------------------------------------------------"

  set_eval_semantic7_mode
  export CRAFTER_OUTPUT_DIR="$eval_dir"
  export TESTER_EVAL_CHECKPOINT="$ckpt_dir"
  export TESTER_REF_CHECKPOINT="$BASE_CKPT"
  export TESTER_EVAL_STEPS="$EVAL_STEPS"
  export TESTER_EVAL_THRESHOLD_Q="0.99"

  # Keep evaluation reward clean:
  # - no tester reward shaping
  # - no RND intrinsic reward during eval
  # - no RND predictor updates during eval
  env -u LD_LIBRARY_PATH \
    CRAFTER_TESTER_REWARD=0 \
    CRAFTER_USE_RND=0 \
    CRAFTER_RND_UPDATE=0 \
    "$PYTHON_BIN" "$MAIN_PY" \
    "${COMMON_EVAL_ARGS[@]}" \
    --logdir "$eval_dir" \
    --run.from_checkpoint "$ckpt_dir"

  echo "[DONE] $name"
  echo
}

# =========================================================
# 1) Existing Dreamer: bug env 100k adaptation + 100k eval
#    - task reward only
# =========================================================
run_case "dreamer_bug_adapt_100k" "$BASE_CKPT" \
  CRAFTER_TESTER_REWARD=0 \
  CRAFTER_USE_RND=0

# =========================================================
# 2) Dreamer + RND: bug env 100k adaptation + 100k eval
#    - task reward + RND intrinsic reward only
# =========================================================
run_case "dreamer_rnd_bug_adapt_100k" "$RND_CKPT" \
  CRAFTER_TESTER_REWARD=0 \
  CRAFTER_USE_RND=1 \
  CRAFTER_RND_ALPHA=0.05 \
  CRAFTER_RND_UPDATE=1 \
  CRAFTER_RND_NORM=1 \
  CRAFTER_RND_CLIP=5.0

echo "===================================================="
echo "ALL RUNS DONE"
echo "results saved under: $LOGROOT"
echo "===================================================="