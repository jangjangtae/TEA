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

RUN_TAG="semantic_eval_100k_$(date +%Y%m%d_%H%M%S)"

# =========================================================
# 헬퍼
# =========================================================
find_latest_ckpt_dir() {
  local train_dir="$1"
  find "$train_dir/ckpt" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

run_eval() {
  local name="$1"
  local ckpt_dir="$2"
  local outdir="$ROOT/${name}_${RUN_TAG}"

  mkdir -p "$outdir"

  echo "===================================================="
  echo "[EVAL] $name"
  echo "checkpoint : $ckpt_dir"
  echo "logdir     : $outdir"
  echo "===================================================="

  # legacy low-level fault 비활성화
  export CRAFTER_FAULT_SAMPLER=0
  export CRAFTER_FAULT=0
  unset CRAFTER_ACTION_SUBTYPES || true
  unset CRAFTER_CONTEXT_SUBTYPES || true
  unset CRAFTER_REWARD_SUBTYPES || true
  unset CRAFTER_TERMINATION_SUBTYPES || true
  unset CRAFTER_FAULT_PROFILE || true
  unset CRAFTER_FAULT_FAMILIES || true
  unset CRAFTER_TRACE_PATH || true

  # semantic high-level fault 7개만 활성화
  export CRAFTER_SEMANTIC_FAULT_SAMPLER=1
  export CRAFTER_SEMANTIC_FAULT_PROFILE=eval_holdout
  export CRAFTER_SEMANTIC_FAULT_EP_PROB=0.5
  export CRAFTER_SEMANTIC_FAULT_VERBOSE=0
  export CRAFTER_SEMANTIC_SUBTYPES=tool_collect_desync_on_upgrade,craft_result_missing_on_retry,station_place_ghost_on_relocate,achievement_unlock_missing_after_valid_progress,station_usable_flag_broken_after_relocate,recipe_precondition_mischeck_on_retry,delayed_inventory_desync_after_station_use

  # env-side 로그를 전부 같은 폴더에 저장
  export CRAFTER_OUTPUT_DIR="$outdir"

  export TESTER_EVAL_CHECKPOINT="$ckpt_dir"
  export TESTER_REF_CHECKPOINT="$REF_CKPT"
  export TESTER_EVAL_STEPS="$EVAL_STEPS"
  export TESTER_EVAL_THRESHOLD_Q="$THRESH_Q"

  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$MAIN_PY" \
    --script tester_eval \
    --configs crafter \
    --logdir "$outdir" \
    --run.from_checkpoint "$ckpt_dir"

  echo "[DONE] $name"
  echo
}

# =========================================================
# baseline 먼저
# =========================================================
run_eval "baseline" "$REF_CKPT"

# =========================================================
# train-only 결과들 평가
# =========================================================
for train_name in \
  a1_task_only \
  a2_task_bug \
  a3_task_cov \
  a4_task_bug_cov \
  a5_full_v17
do
  train_dir="$ROOT/$train_name"

  if [[ ! -d "$train_dir" ]]; then
    echo "[SKIP] missing dir: $train_dir"
    continue
  fi

  ckpt_dir="$(find_latest_ckpt_dir "$train_dir")"
  if [[ -z "${ckpt_dir:-}" ]]; then
    echo "[SKIP] no ckpt found under: $train_dir/ckpt"
    continue
  fi

  run_eval "$train_name" "$ckpt_dir"
done

echo "===================================================="
echo "ALL DONE"
echo "Saved under: $ROOT/*_${RUN_TAG}"
echo "===================================================="