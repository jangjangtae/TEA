#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/logdir/ablation_20260324_181013"
PROJECT_ROOT="$HOME/dreamerv3"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"
PYTHON_BIN="python"

REF_CKPT="/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487"
EVAL_STEPS="${EVAL_STEPS:-50000}"
THRESH_Q="${THRESH_Q:-0.99}"

RUN_TAG="semantic_eval_$(date +%Y%m%d_%H%M%S)"

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

  # legacy low-level fault 완전 비활성화
  export CRAFTER_FAULT_SAMPLER=0
  export CRAFTER_FAULT=0
  unset CRAFTER_ACTION_SUBTYPES
  unset CRAFTER_CONTEXT_SUBTYPES
  unset CRAFTER_REWARD_SUBTYPES
  unset CRAFTER_TERMINATION_SUBTYPES
  unset CRAFTER_FAULT_PROFILE
  unset CRAFTER_FAULT_FAMILIES

  # semantic high-level fault만 활성화
  export CRAFTER_SEMANTIC_FAULT_SAMPLER=1
  export CRAFTER_SEMANTIC_FAULT_PROFILE=eval_holdout
  export CRAFTER_SEMANTIC_FAULT_EP_PROB=0.5
  export CRAFTER_SEMANTIC_FAULT_VERBOSE=0
  export CRAFTER_SEMANTIC_SUBTYPES=tool_collect_desync_on_upgrade,craft_result_missing_on_retry,station_place_ghost_on_relocate,achievement_unlock_missing_after_valid_progress,station_usable_flag_broken_after_relocate,recipe_precondition_mischeck_on_retry,delayed_inventory_desync_after_station_use

  # 출력 파일을 run 폴더 안으로 고정
  export CRAFTER_OUTPUT_DIR="$outdir"
  unset CRAFTER_TRACE_PATH

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

# baseline
run_eval "baseline" "$REF_CKPT"

# ablation models
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
    echo "[SKIP] no ckpt: $train_dir/ckpt"
    continue
  fi

  run_eval "$train_name" "$ckpt_dir"
done

echo "===================================================="
echo "ALL DONE"
echo "Saved under: $ROOT/*_${RUN_TAG}"
echo "===================================================="