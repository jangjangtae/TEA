#!/usr/bin/env bash
set -euo pipefail

# Multi-seed MiniGrid fault-seeking queue.
#
# The queue first trains a clean MiniGrid reference model for each seed, then
# adapts that seed on the benchmark_train fault split and evaluates on clean,
# seen, holdout, and sparse splits. Replay buffers are pruned after each variant
# by default so the run can be left unattended over a weekend.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"
ANALYZE_PY="$PROJECT_ROOT/dreamerv3/analyze_minigrid_multiseed.py"

ROOT="${ROOT:-$HOME/logdir/minigrid_multiseed_fault_$(date +%Y%m%d_%H%M%S)}"
CONFIGS="${CONFIGS:-minigrid}"

SEEDS="${SEEDS:-0 1 2}"
VARIANTS="${VARIANTS:-taskonly dense_beta02 excess_delta_p95_beta02 contextual_excess_delta_beta02}"
EVAL_SPLITS="${EVAL_SPLITS:-clean seen holdout sparse}"

CLEAN_TRAIN_STEPS="${CLEAN_TRAIN_STEPS:-500000}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
TRAIN_MILESTONES="${TRAIN_MILESTONES:-100000 300000 500000}"
TRAIN_ENVS="${TRAIN_ENVS:-16}"
TRAIN_RATIO="${TRAIN_RATIO:-128}"
REPLAY_SIZE="${REPLAY_SIZE:-50000}"
SAVE_EVERY="${SAVE_EVERY:-600}"

BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-20000}"
ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-20000}"
SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-40000}"
MILESTONE_EVAL_STEPS="${MILESTONE_EVAL_STEPS:-10000}"
MILESTONE_SPARSE_EVAL_STEPS="${MILESTONE_SPARSE_EVAL_STEPS:-20000}"

BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
REPORT_LENGTH="${REPORT_LENGTH:-32}"
EVAL_ENVS="${EVAL_ENVS:-1}"

JAX_PLATFORM="${JAX_PLATFORM:-cuda}"
JAX_PREALLOC="${JAX_PREALLOC:-True}"
JAX_COMPUTE_DTYPE="${JAX_COMPUTE_DTYPE:-float32}"

MIN_FREE_GB="${MIN_FREE_GB:-25}"
PRUNE_REPLAY_AFTER_TRAIN="${PRUNE_REPLAY_AFTER_TRAIN:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
RUN_BASE_EVALS="${RUN_BASE_EVALS:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

cd "$PROJECT_ROOT"

if [[ -d "$PROJECT_ROOT/.deps/minigrid_pkgs" ]]; then
  export PYTHONPATH="$PROJECT_ROOT/.deps/minigrid_pkgs${PYTHONPATH:+:$PYTHONPATH}"
fi

export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-off}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
SYSTEM_CUDA_LIBS="${SYSTEM_CUDA_LIBS:-/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib/python3/dist-packages/tensorflow}"
export LD_LIBRARY_PATH="$SYSTEM_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -z "${XLA_FLAGS:-}" && -d /usr/lib/cuda/nvvm/libdevice ]]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"
fi
if [[ -f /usr/lib/cuda/nvvm/libdevice/libdevice.10.bc && ! -e "$PROJECT_ROOT/libdevice.10.bc" ]]; then
  ln -s /usr/lib/cuda/nvvm/libdevice/libdevice.10.bc "$PROJECT_ROOT/libdevice.10.bc"
fi

mkdir -p "$ROOT"
LAUNCHER_LOG="$ROOT/launcher.log"
STATUS_FILE="$ROOT/status.tsv"
MILESTONE_FILE="$ROOT/milestones.tsv"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

stamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

record_status() {
  local seed="$1"
  local name="$2"
  local status="$3"
  echo -e "$(stamp)\tseed=${seed}\t${name}\t${status}" | tee -a "$STATUS_FILE"
}

free_gb() {
  df -Pk "$ROOT" | awk 'NR == 2 {printf "%.1f", $4 / 1024 / 1024}'
}

check_disk() {
  local available
  available="$(free_gb)"
  echo "[disk] available=${available}GB min=${MIN_FREE_GB}GB root=$ROOT"
  "$PYTHON_BIN" - "$available" "$MIN_FREE_GB" <<'PY'
import sys
available = float(sys.argv[1])
minimum = float(sys.argv[2])
if available < minimum:
  raise SystemExit(f"Disk guard: only {available:.1f}GB free, need {minimum:.1f}GB")
PY
}

root_usage() {
  du -sh "$ROOT" 2>/dev/null || true
}

run_job() {
  local seed="$1"
  local name="$2"
  shift 2
  local seed_root="$ROOT/seed_${seed}"
  local log_file="$seed_root/${name}.log"
  mkdir -p "$seed_root"

  check_disk
  record_status "$seed" "$name" START
  echo "====================================================" | tee -a "$log_file"
  echo "[$name] $(stamp) seed=$seed" | tee -a "$log_file"
  echo "Log: $log_file" | tee -a "$log_file"
  echo "====================================================" | tee -a "$log_file"

  set +e
  "$@" 2>&1 | tee -a "$log_file"
  local code="${PIPESTATUS[0]}"
  set -e

  if [[ "$code" -eq 0 ]]; then
    record_status "$seed" "$name" DONE
  else
    record_status "$seed" "$name" "FAILED:${code}"
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      echo "[queue] stopping after failure in $name" >&2
      exit "$code"
    fi
  fi
  root_usage
}

latest_ckpt() {
  local root="$1"
  local ckpt_root="$root/ckpt"
  if [[ -f "$ckpt_root/latest" ]]; then
    local latest
    latest="$(cat "$ckpt_root/latest")"
    if [[ -d "$ckpt_root/$latest" ]]; then
      echo "$ckpt_root/$latest"
      return 0
    fi
  fi
  find "$ckpt_root" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

clean_fault_env() {
  export MINIGRID_FAULT=0
  unset MINIGRID_FAULT_PROFILE || true
  unset MINIGRID_FAULT_TYPE || true
  unset MINIGRID_FAULT_EP_PROB || true
  unset MINIGRID_FAULT_MANIFEST_PROB || true
}

fault_env() {
  local profile="$1"
  local ep_prob="$2"
  local manifest_prob="$3"
  export MINIGRID_FAULT=1
  export MINIGRID_FAULT_PROFILE="$profile"
  export MINIGRID_FAULT_EP_PROB="$ep_prob"
  export MINIGRID_FAULT_MANIFEST_PROB="$manifest_prob"
}

split_profile() {
  local split="$1"
  case "$split" in
    clean) echo "" ;;
    diagnostic) echo "diagnostic" ;;
    seen) echo "benchmark_seen" ;;
    holdout) echo "benchmark_holdout" ;;
    sparse) echo "benchmark_sparse" ;;
    *) echo "Unknown split: $split" >&2; return 1 ;;
  esac
}

split_probs() {
  local split="$1"
  case "$split" in
    sparse) echo "0.1 1.0" ;;
    seen|holdout|diagnostic) echo "0.5 1.0" ;;
    clean) echo "" ;;
    *) echo "Unknown split: $split" >&2; return 1 ;;
  esac
}

split_steps() {
  local split="$1"
  local milestone="${2:-}"
  local final=1
  if [[ -n "$milestone" && "$milestone" -lt "$TRAIN_STEPS" ]]; then
    final=0
  fi
  if [[ "$split" == "sparse" ]]; then
    if (( final )); then echo "$SPARSE_EVAL_STEPS"; else echo "$MILESTONE_SPARSE_EVAL_STEPS"; fi
  else
    if (( final )); then echo "$ADAPT_EVAL_STEPS"; else echo "$MILESTONE_EVAL_STEPS"; fi
  fi
}

variant_params() {
  local variant="$1"
  case "$variant" in
    taskonly)
      echo "True dense 0.0 p95"
      ;;
    dense_beta02)
      echo "False dense 0.2 p95"
      ;;
    excess_delta_p95_beta02)
      echo "False excess_delta_threshold 0.2 p95"
      ;;
    contextual_excess_delta_beta02)
      echo "False excess_delta_threshold 0.2 context_p95"
      ;;
    contextual_excess_delta_beta01)
      echo "False excess_delta_threshold 0.1 context_p95"
      ;;
    contextual_excess_delta_beta05)
      echo "False excess_delta_threshold 0.5 context_p95"
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      return 1
      ;;
  esac
}

run_clean_train() {
  local seed="$1"
  local outdir="$ROOT/seed_${seed}/clean_train/train"
  clean_fault_env
  "$PYTHON_BIN" "$MAIN_PY" \
    --script train \
    --configs $CONFIGS \
    --logdir "$outdir" \
    --seed "$seed" \
    --run.steps "$CLEAN_TRAIN_STEPS" \
    --run.envs "$TRAIN_ENVS" \
    --run.train_ratio "$TRAIN_RATIO" \
    --run.log_every 60 \
    --run.report_every 120 \
    --run.save_every "$SAVE_EVERY" \
    --run.debug False \
    --replay.size "$REPLAY_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --batch_length "$BATCH_LENGTH" \
    --report_length "$REPORT_LENGTH" \
    --jax.platform "$JAX_PLATFORM" \
    --jax.prealloc "$JAX_PREALLOC" \
    --jax.compute_dtype "$JAX_COMPUTE_DTYPE" \
    --jax.compilation_cache False \
    --jax.autotune 0 \
    --jax.deterministic False \
    --jax.prefetch False \
    --jax.profiler False \
    --jax.donate_train False \
    --jax.expect_devices 0
}

run_eval() {
  local seed="$1"
  local name="$2"
  local ckpt="$3"
  local ref_ckpt="$4"
  local steps="$5"
  local profile="$6"
  local ep_prob="$7"
  local manifest_prob="$8"
  local stats="${9:-}"
  local norm_mode="${10:-p95}"
  local outdir="${EVAL_OUTPUT_ROOT:-$ROOT}/seed_${seed}/$name"

  if [[ -n "$profile" ]]; then
    fault_env "$profile" "$ep_prob" "$manifest_prob"
  else
    clean_fault_env
  fi

  "$PYTHON_BIN" "$MAIN_PY" \
    --script eval_only \
    --configs $CONFIGS \
    --logdir "$outdir" \
    --seed "$seed" \
    --run.from_checkpoint "$ckpt" \
    --run.steps "$steps" \
    --run.envs "$EVAL_ENVS" \
    --run.debug True \
    --run.log_every 60 \
    --batch_size "$BATCH_SIZE" \
    --batch_length "$BATCH_LENGTH" \
    --report_length "$REPORT_LENGTH" \
    --jax.platform "$JAX_PLATFORM" \
    --jax.prealloc "$JAX_PREALLOC" \
    --jax.compute_dtype "$JAX_COMPUTE_DTYPE" \
    --jax.compilation_cache False \
    --jax.autotune 0 \
    --jax.deterministic False \
    --jax.prefetch False \
    --jax.profiler False \
    --jax.donate_train False \
    --jax.expect_devices 0 \
    --fault.enabled True \
    --fault.ref_ckpt "$ref_ckpt" \
    --fault.norm_stats "$stats" \
    --fault.norm_mode "$norm_mode" \
    --fault.score_source latent_reward \
    --fault.use_kl_bound False \
    --fault.log_only True \
    --fault.trace fault_score_trace.jsonl
}

make_norm_stats() {
  local trace="$1"
  local out="$2"
  "$PYTHON_BIN" - "$trace" "$out" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

trace = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
with trace.open() as f:
  for line in f:
    if line.strip():
      rows.append(json.loads(line))

def values(*keys):
  vals = []
  for row in rows:
    for key in keys:
      if key in row:
        vals.append(float(row.get(key, 0.0)))
        break
    else:
      vals.append(0.0)
  return np.asarray(vals, np.float64)

def summarize(prefix, arr):
  arr = arr[np.isfinite(arr)]
  if arr.size == 0:
    arr = np.asarray([0.0])
  return {
      f"{prefix}_mean": float(arr.mean()),
      f"{prefix}_std": float(arr.std()),
      f"{prefix}_p90": float(np.percentile(arr, 90)),
      f"{prefix}_p95": float(np.percentile(arr, 95)),
      f"{prefix}_p99": float(np.percentile(arr, 99)),
      f"{prefix}_count": int(arr.size),
  }

fault = values("fault_score_raw", "fault/fault_score_raw", "log/ref_fault_score_raw")
data = {}
data.update(summarize("latent_kl", values("latent_kl_surprise", "fault/latent_kl_surprise", "log/ref_latent_kl_surprise")))
data.update(summarize("reward_error", values("reward_prediction_error", "fault/reward_prediction_error", "log/ref_reward_prediction_error")))
data.update(summarize("fault_score", fault))
context_scores = {}
for row, score in zip(rows, fault):
  action = int(row.get("action", 0))
  inv = int(row.get("inventory_bucket", 0))
  tile = int(row.get("nearby_tile", 0))
  stage = int(row.get("achievement_stage", 0))
  mob = int(row.get("nearby_mob", 0))
  keys = (
      f"full:a={action}|i={inv}|t={tile}|g={stage}|m={mob}",
      f"action_stage:a={action}|g={stage}",
      f"action:a={action}",
      "global",
  )
  for key in keys:
    context_scores.setdefault(key, []).append(score)
data["context_schema"] = {
    "version": 1,
    "fields": ["action", "inventory_bucket", "nearby_tile", "achievement_stage", "nearby_mob"],
    "fallback_order": ["full", "action_stage", "action", "global"],
}
data["context_stats"] = {
    key: summarize("fault_score", np.asarray(vals, np.float64))
    for key, vals in sorted(context_scores.items())
}
data["steps"] = int(len(rows))
out.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
print("Wrote norm stats:", out)
PY
}

run_train_variant() {
  local seed="$1"
  local variant="$2"
  local ref_ckpt="$3"
  local stats="$4"
  local target_steps="${5:-$TRAIN_STEPS}"
  local params log_only reward_mode beta norm_mode
  params="$(variant_params "$variant")"
  read -r log_only reward_mode beta norm_mode <<< "$params"
  local outdir="$ROOT/seed_${seed}/train_${variant}/train"

  fault_env "benchmark_train" "0.5" "1.0"

  "$PYTHON_BIN" "$MAIN_PY" \
    --script train \
    --configs $CONFIGS \
    --logdir "$outdir" \
    --seed "$seed" \
    --run.from_checkpoint "$ref_ckpt" \
    --run.steps "$target_steps" \
    --run.envs "$TRAIN_ENVS" \
    --run.train_ratio "$TRAIN_RATIO" \
    --run.log_every 60 \
    --run.report_every 120 \
    --run.save_every "$SAVE_EVERY" \
    --run.debug False \
    --replay.size "$REPLAY_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --batch_length "$BATCH_LENGTH" \
    --report_length "$REPORT_LENGTH" \
    --jax.platform "$JAX_PLATFORM" \
    --jax.prealloc "$JAX_PREALLOC" \
    --jax.compute_dtype "$JAX_COMPUTE_DTYPE" \
    --jax.compilation_cache False \
    --jax.autotune 0 \
    --jax.deterministic False \
    --jax.prefetch False \
    --jax.profiler False \
    --jax.donate_train False \
    --jax.expect_devices 0 \
    --fault.enabled True \
    --fault.ref_ckpt "$ref_ckpt" \
    --fault.norm_stats "$stats" \
    --fault.log_only "$log_only" \
    --fault.beta "$beta" \
    --fault.reward_mode "$reward_mode" \
    --fault.norm_mode "$norm_mode" \
    --fault.score_source latent_reward \
    --fault.use_kl_bound False \
    --fault.reward_threshold 1.0 \
    --fault.reward_delta_threshold 0.5 \
    --fault.clip 2.0 \
    --fault.reward_gate none \
    --fault.use_reward_error True \
    --fault.w_reward 0.0 \
    --fault.trace fault_score_trace.jsonl
}

prune_replay() {
  local train_dir="$1"
  if [[ "$PRUNE_REPLAY_AFTER_TRAIN" == "1" && -d "$train_dir/replay" ]]; then
    echo "[prune] removing completed replay: $train_dir/replay"
    rm -rf "$train_dir/replay"
  fi
}

archive_milestone_ckpt() {
  local seed="$1"
  local variant="$2"
  local milestone="$3"
  local source="$4"
  local target="$ROOT/checkpoints/seed_${seed}/${variant}/step_${milestone}"
  if [[ ! -f "$target/done" ]]; then
    mkdir -p "$target"
    cp -a "$source/." "$target/"
    touch "$target/done"
  fi
  local actual_step
  actual_step="$($PYTHON_BIN - "$target/step.pkl" <<'PY'
import pickle
import sys
with open(sys.argv[1], "rb") as f:
  print(int(pickle.load(f)))
PY
)"
  if [[ ! -f "$MILESTONE_FILE" ]]; then
    printf 'seed\tvariant\ttarget_step\tactual_step\tcheckpoint\n' > "$MILESTONE_FILE"
  fi
  if ! awk -F '\t' -v seed="$seed" -v variant="$variant" -v target="$milestone" \
      'NR > 1 && $1 == seed && $2 == variant && $3 == target {found=1} END {exit !found}' \
      "$MILESTONE_FILE"; then
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$seed" "$variant" "$milestone" "$actual_step" "$target" >> "$MILESTONE_FILE"
  fi
  printf '%s\n' "$target"
}

run_variant_evals() {
  local seed="$1"
  local variant="$2"
  local ckpt="$3"
  local ref_ckpt="$4"
  local stats="$5"
  local milestone="${6:-}"
  local params norm_mode
  params="$(variant_params "$variant")"
  read -r _ _ _ norm_mode <<< "$params"
  local previous_eval_root="${EVAL_OUTPUT_ROOT:-}"
  if [[ -n "$milestone" ]]; then
    EVAL_OUTPUT_ROOT="$ROOT/milestone_${milestone}"
  fi
  for split in $EVAL_SPLITS; do
    local profile probs ep_prob manifest_prob steps eval_name job_name
    profile="$(split_profile "$split")"
    if [[ -n "$profile" ]]; then
      probs="$(split_probs "$split")"
      read -r ep_prob manifest_prob <<< "$probs"
    else
      ep_prob=""
      manifest_prob=""
    fi
    steps="$(split_steps "$split" "$milestone")"
    eval_name="eval_${variant}_${split}"
    job_name="$eval_name"
    if [[ -n "$milestone" ]]; then
      job_name="${eval_name}_at_${milestone}"
    fi
    run_job "$seed" "$job_name" \
      run_eval "$seed" "$eval_name" "$ckpt" "$ref_ckpt" "$steps" "$profile" "$ep_prob" "$manifest_prob" "$stats" "$norm_mode"
  done
  if [[ -n "$previous_eval_root" ]]; then
    EVAL_OUTPUT_ROOT="$previous_eval_root"
  else
    unset EVAL_OUTPUT_ROOT || true
  fi
}

validate_milestones() {
  local previous=0
  local milestone
  for milestone in $TRAIN_MILESTONES; do
    if (( milestone <= previous || milestone > TRAIN_STEPS )); then
      echo "Invalid TRAIN_MILESTONES='$TRAIN_MILESTONES' for TRAIN_STEPS=$TRAIN_STEPS" >&2
      return 1
    fi
    previous="$milestone"
  done
  if [[ -n "$TRAIN_MILESTONES" && "$previous" -ne "$TRAIN_STEPS" ]]; then
    echo "Final milestone must equal TRAIN_STEPS ($TRAIN_STEPS): $TRAIN_MILESTONES" >&2
    return 1
  fi
}

echo "===================================================="
echo "[MiniGrid multi-seed fault queue]"
echo "root             : $ROOT"
echo "configs          : $CONFIGS"
echo "seeds            : $SEEDS"
echo "variants         : $VARIANTS"
echo "splits           : $EVAL_SPLITS"
echo "clean steps      : $CLEAN_TRAIN_STEPS"
echo "adapt steps      : $TRAIN_STEPS"
echo "milestones       : ${TRAIN_MILESTONES:-disabled}"
echo "train ratio      : $TRAIN_RATIO"
echo "replay size      : $REPLAY_SIZE"
echo "min free GB      : $MIN_FREE_GB"
echo "prune replay     : $PRUNE_REPLAY_AFTER_TRAIN"
echo "jax platform     : $JAX_PLATFORM"
echo "python           : $PYTHON_BIN"
echo "PYTHONPATH       : ${PYTHONPATH:-}"
echo "LD_LIBRARY       : ${LD_LIBRARY_PATH:-}"
echo "XLA_FLAGS        : ${XLA_FLAGS:-}"
echo "===================================================="

validate_milestones

for seed in $SEEDS; do
  seed_root="$ROOT/seed_${seed}"
  mkdir -p "$seed_root"

  run_job "$seed" "clean_train" run_clean_train "$seed"
  REF_CKPT="$(latest_ckpt "$seed_root/clean_train/train")"
  if [[ -z "$REF_CKPT" ]]; then
    echo "Missing clean checkpoint for seed=$seed" >&2
    exit 1
  fi
  prune_replay "$seed_root/clean_train/train"

  if [[ "$RUN_BASE_EVALS" == "1" ]]; then
    run_job "$seed" "base_clean_eval" \
      run_eval "$seed" "base_clean_eval" "$REF_CKPT" "$REF_CKPT" "$BASE_EVAL_STEPS" "" "" ""
    run_job "$seed" "base_seen_eval" \
      run_eval "$seed" "base_seen_eval" "$REF_CKPT" "$REF_CKPT" "$BASE_EVAL_STEPS" "benchmark_seen" "0.5" "1.0"
    run_job "$seed" "base_holdout_eval" \
      run_eval "$seed" "base_holdout_eval" "$REF_CKPT" "$REF_CKPT" "$BASE_EVAL_STEPS" "benchmark_holdout" "0.5" "1.0"
    run_job "$seed" "base_sparse_eval" \
      run_eval "$seed" "base_sparse_eval" "$REF_CKPT" "$REF_CKPT" "$SPARSE_EVAL_STEPS" "benchmark_sparse" "0.1" "1.0"
  fi

  STATS_FILE="$seed_root/minigrid_clean_fault_stats.json"
  if [[ ! -f "$STATS_FILE" ]]; then
    if [[ ! -f "$seed_root/base_clean_eval/fault_score_trace.jsonl" ]]; then
      run_job "$seed" "base_clean_eval_for_stats" \
        run_eval "$seed" "base_clean_eval" "$REF_CKPT" "$REF_CKPT" "$BASE_EVAL_STEPS" "" "" ""
    fi
    run_job "$seed" "make_norm_stats" \
      make_norm_stats "$seed_root/base_clean_eval/fault_score_trace.jsonl" "$STATS_FILE"
  fi

  for variant in $VARIANTS; do
    if [[ -n "$TRAIN_MILESTONES" ]]; then
      for milestone in $TRAIN_MILESTONES; do
        run_job "$seed" "train_${variant}_to_${milestone}" \
          run_train_variant "$seed" "$variant" "$REF_CKPT" "$STATS_FILE" "$milestone"
        variant_ckpt="$(latest_ckpt "$seed_root/train_${variant}/train")"
        if [[ -z "$variant_ckpt" ]]; then
          echo "Missing checkpoint: seed=$seed variant=$variant milestone=$milestone" >&2
          exit 1
        fi
        milestone_ckpt="$(archive_milestone_ckpt "$seed" "$variant" "$milestone" "$variant_ckpt")"
        run_variant_evals "$seed" "$variant" "$milestone_ckpt" "$REF_CKPT" "$STATS_FILE" "$milestone"
      done
    else
      run_job "$seed" "train_${variant}" \
        run_train_variant "$seed" "$variant" "$REF_CKPT" "$STATS_FILE"
      variant_ckpt="$(latest_ckpt "$seed_root/train_${variant}/train")"
      if [[ -z "$variant_ckpt" ]]; then
        echo "Missing variant checkpoint for seed=$seed variant=$variant" >&2
        exit 1
      fi
      run_variant_evals "$seed" "$variant" "$variant_ckpt" "$REF_CKPT" "$STATS_FILE"
    fi
    prune_replay "$seed_root/train_${variant}/train"
  done
done

if [[ "$RUN_ANALYSIS" == "1" && -f "$ANALYZE_PY" ]]; then
  if [[ -n "$TRAIN_MILESTONES" ]]; then
    for milestone in $TRAIN_MILESTONES; do
      run_job "all" "analysis_${milestone}" \
        "$PYTHON_BIN" "$ANALYZE_PY" \
          --root "$ROOT/milestone_${milestone}" \
          --outdir "$ROOT/analysis/milestone_${milestone}" \
          --baseline taskonly --eval-only
    done
  else
    run_job "all" "analysis" \
      "$PYTHON_BIN" "$ANALYZE_PY" \
        --root "$ROOT" \
        --outdir "$ROOT/analysis" \
        --baseline taskonly --eval-only
  fi
fi

echo "===================================================="
echo "[MiniGrid multi-seed fault queue] DONE"
echo "root    : $ROOT"
echo "status  : $STATUS_FILE"
echo "log     : $LAUNCHER_LOG"
echo "analysis: $ROOT/analysis"
echo "===================================================="
