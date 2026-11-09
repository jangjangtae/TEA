#!/usr/bin/env bash
set -euo pipefail

# Evaluation-only DreamerV3 adaptation of the ICML 2025 KL-bound detector.
# It compares the current posterior-prior KL against the paper's threshold-free
# bound on identical Craftax policies and fault splits.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"
ANALYZE_PY="$PROJECT_ROOT/dreamerv3/analyze_craftax_multiseed.py"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-$HOME/logdir/craftax_kl_bound_probe_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-0}"
SPLITS="${SPLITS:-clean seen holdout sparse}"
EVAL_STEPS="${EVAL_STEPS:-30000}"
SPARSE_STEPS="${SPARSE_STEPS:-60000}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"

cd "$PROJECT_ROOT"

if [[ -d "$PROJECT_ROOT/.deps/craftax_pkgs" ]]; then
  export PYTHONPATH="$PROJECT_ROOT/.deps/craftax_pkgs${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONFAULTHANDLER=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib/python3/dist-packages/tensorflow${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -z "${XLA_FLAGS:-}" && -d /usr/lib/cuda/nvvm/libdevice ]]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"
fi

latest_ckpt() {
  local root="$1/ckpt"
  if [[ -f "$root/latest" ]]; then
    local name
    name="$(cat "$root/latest")"
    [[ -d "$root/$name" ]] && { echo "$root/$name"; return; }
  fi
  find "$root" -mindepth 1 -maxdepth 1 -type d | sort | tail -1
}

REF_CKPT="$(latest_ckpt "$TRAIN_ROOT/train")"
[[ -n "$REF_CKPT" ]] || { echo "Missing clean checkpoint" >&2; exit 1; }
mkdir -p "$ROOT"
STATUS="$ROOT/status.tsv"
exec > >(tee -a "$ROOT/launcher.log") 2>&1

check_disk() {
  local available
  available="$(df -Pk "$ROOT" | awk 'NR == 2 {printf "%.1f", $4 / 1024 / 1024}')"
  echo "[disk] available=${available}GB min=${MIN_FREE_GB}GB root=$ROOT"
  "$PYTHON_BIN" - "$available" "$MIN_FREE_GB" <<'PY'
import sys
available, minimum = map(float, sys.argv[1:])
if available < minimum:
  raise SystemExit(
      f"Disk guard: only {available:.1f}GB free, need {minimum:.1f}GB")
PY
}

set_split() {
  local split="$1"
  export CRAFTAX_FAULT=0 CRAFTAX_FAULT_SAMPLER=0
  case "$split" in
    clean) unset CRAFTAX_FAULT_PROFILE || true ;;
    seen) export CRAFTAX_FAULT_SAMPLER=1 CRAFTAX_FAULT_PROFILE=benchmark_seen CRAFTAX_FAULT_EP_PROB=0.5 ;;
    holdout) export CRAFTAX_FAULT_SAMPLER=1 CRAFTAX_FAULT_PROFILE=benchmark_holdout CRAFTAX_FAULT_EP_PROB=0.5 ;;
    sparse) export CRAFTAX_FAULT_SAMPLER=1 CRAFTAX_FAULT_PROFILE=benchmark_sparse CRAFTAX_FAULT_EP_PROB=0.1 ;;
  esac
  export CRAFTAX_FAULT_MANIFEST_PROB=1.0
}

run_eval() {
  local seed="$1" method="$2" split="$3" source="$4" use_bound="$5"
  local steps="$EVAL_STEPS"
  [[ "$split" == sparse ]] && steps="$SPARSE_STEPS"
  local out="$ROOT/seed_${seed}/eval_${method}_${split}"
  mkdir -p "$out"
  check_disk
  set_split "$split"
  export CRAFTAX_FAULT_SEED="$seed"
  echo -e "$(date '+%F %T')\tseed=${seed}\teval_${method}_${split}\tSTART" >> "$STATUS"
  set +e
  "$PYTHON_BIN" "$MAIN_PY" \
    --script eval_only --configs craftax size1m --logdir "$out" --seed "$seed" \
    --run.from_checkpoint "$REF_CKPT" --run.steps "$steps" --run.envs 1 \
    --run.debug True --run.log_every 60 --env.craftax.platform "" \
    --batch_size 16 --batch_length 64 --report_length 32 \
    --jax.platform cuda --jax.prealloc True --jax.compute_dtype float32 \
    --jax.compilation_cache False --jax.autotune 0 --jax.deterministic False \
    --jax.prefetch False --jax.profiler False --jax.donate_train False \
    --jax.expect_devices 0 --fault.enabled True --fault.ref_ckpt "$REF_CKPT" \
    --fault.log_only True --fault.norm_mode none \
    --fault.score_source "$source" --fault.use_kl_bound "$use_bound" \
    --fault.use_latent_kl True --fault.use_reward_error False \
    --fault.w_reward 0.0 --fault.suspicious_threshold 0.0 \
    --fault.trace fault_score_trace.jsonl
  local code=$?
  set -e
  if (( code != 0 )); then
    echo -e "$(date '+%F %T')\tseed=${seed}\teval_${method}_${split}\tFAILED:${code}" >> "$STATUS"
    return "$code"
  fi
  echo -e "$(date '+%F %T')\tseed=${seed}\teval_${method}_${split}\tDONE" >> "$STATUS"
}

echo "KL-bound probe root: $ROOT"
echo "Reference checkpoint: $REF_CKPT"
for seed in $SEEDS; do
  for split in $SPLITS; do
    run_eval "$seed" latent_kl_reference "$split" latent_reward False
    run_eval "$seed" kl_bound_reference "$split" kl_bound True
  done
done

"$PYTHON_BIN" "$ANALYZE_PY" --root "$ROOT" --outdir "$ROOT/analysis" \
  --baseline latent_kl_reference --eval-only
echo "KL-bound probe DONE: $ROOT"
