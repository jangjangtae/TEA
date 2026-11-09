#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# 기본 설정
# =========================================================
ABLATION_ROOT="$HOME/logdir/ablation_20260324_181013"
PROJECT_ROOT="$HOME/dreamerv3"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"
PYTHON_BIN="python"

REF_CKPT="/home/railab/logdir/dreamer_clean/20260303T112635/ckpt/20260305T131501F807487"
EVAL_STEPS="${EVAL_STEPS:-50000}"
THRESH_Q="${THRESH_Q:-0.99}"

# 결과 저장용 태그
RUN_TAG="newholdout_$(date +%Y%m%d_%H%M%S)"

# =========================================================
# 헬퍼 함수
# =========================================================
find_latest_ckpt_dir() {
  local train_dir="$1"
  find "$train_dir/ckpt" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

run_eval() {
  local name="$1"
  local ckpt_dir="$2"
  local outdir="$ABLATION_ROOT/${name}_${RUN_TAG}"

  mkdir -p "$outdir"

  echo "===================================================="
  echo "[EVAL] $name"
  echo "checkpoint : $ckpt_dir"
  echo "logdir     : $outdir"
  echo "===================================================="

  # 새 holdout 환경을 강제로 사용
  export CRAFTER_FAULT_SAMPLER=1
  export CRAFTER_FAULT_PROFILE=eval_holdout
  export CRAFTER_FAULT_EP_PROB=0.5

  # 중요:
  # trace를 logdir 안에 자동 저장하도록 비워둠
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

# =========================================================
# 실행
# =========================================================

# baseline은 clean reference checkpoint 자체를 평가
run_eval "baseline" "$REF_CKPT"

# 예전에 학습한 ablation 모델들 재평가
for train_name in \
  a1_task_only \
  a2_task_bug \
  a3_task_cov \
  a4_task_bug_cov \
  a5_full_v17
do
  train_dir="$ABLATION_ROOT/$train_name"

  if [[ ! -d "$train_dir" ]]; then
    echo "[SKIP] train dir not found: $train_dir"
    continue
  fi

  ckpt_dir="$(find_latest_ckpt_dir "$train_dir")"
  if [[ -z "${ckpt_dir:-}" ]]; then
    echo "[SKIP] checkpoint not found under: $train_dir/ckpt"
    continue
  fi

  run_eval "$train_name" "$ckpt_dir"
done

echo "===================================================="
echo "ALL DONE"
echo "Saved under: $ABLATION_ROOT/*_${RUN_TAG}"
echo "===================================================="