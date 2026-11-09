#!/usr/bin/env bash
set -euo pipefail

# Multi-seed Craftax fault-seeking queue.
#
# This queue is intended for unattended runs: it repeats the same objective
# variants over several seeds, checks disk space before every job, and prunes
# completed replay buffers by default to avoid filling the machine.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
MAIN_PY="$PROJECT_ROOT/dreamerv3/main.py"
ANALYZE_PY="$PROJECT_ROOT/dreamerv3/analyze_craftax_multiseed.py"
ANALYZE_MILESTONES_PY="$PROJECT_ROOT/dreamerv3/analyze_craftax_milestones.py"

TRAIN_ROOT="${TRAIN_ROOT:?Set TRAIN_ROOT to the completed Craftax clean run root.}"
ROOT="${ROOT:-$HOME/logdir/craftax_multiseed_fault_$(date +%Y%m%d_%H%M%S)}"

SEEDS="${SEEDS:-0 1 2}"
VARIANTS="${VARIANTS:-taskonly dense_beta01 excess_p95_beta01 excess_delta_p95_beta01}"
EVAL_SPLITS="${EVAL_SPLITS:-clean seen holdout sparse}"

CONFIGS="${CONFIGS:-craftax size1m}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
TRAIN_MILESTONES="${TRAIN_MILESTONES:-}"
TRAIN_ENVS="${TRAIN_ENVS:-16}"
TRAIN_RATIO="${TRAIN_RATIO:-128}"
REPLAY_SIZE="${REPLAY_SIZE:-100000}"
SAVE_EVERY="${SAVE_EVERY:-600}"

BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-50000}"
ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-30000}"
SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-30000}"
MILESTONE_EVAL_STEPS="${MILESTONE_EVAL_STEPS:-20000}"
MILESTONE_SPARSE_EVAL_STEPS="${MILESTONE_SPARSE_EVAL_STEPS:-40000}"
ADAPTIVE_TASK_TARGET="${ADAPTIVE_TASK_TARGET:-1.45}"
CONSTRAINT_TASK_TARGET="${CONSTRAINT_TASK_TARGET:-1.45}"
CONSTRAINT_WARMUP_EPISODES="${CONSTRAINT_WARMUP_EPISODES:-10}"
RND_ALPHA="${RND_ALPHA:-}"
RND_NORM="${RND_NORM:-1}"
RND_CLIP="${RND_CLIP:-5.0}"
RND_DOWNSAMPLE="${RND_DOWNSAMPLE:-4}"
RND_HIDDEN_DIM="${RND_HIDDEN_DIM:-128}"
RND_OUTPUT_DIM="${RND_OUTPUT_DIM:-128}"
RND_LR="${RND_LR:-0.005}"

BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
REPORT_LENGTH="${REPORT_LENGTH:-32}"
EVAL_ENVS="${EVAL_ENVS:-1}"

JAX_PLATFORM="${JAX_PLATFORM:-cuda}"
JAX_PREALLOC="${JAX_PREALLOC:-True}"
JAX_COMPUTE_DTYPE="${JAX_COMPUTE_DTYPE:-float32}"

MIN_FREE_GB="${MIN_FREE_GB:-40}"
PRUNE_REPLAY_AFTER_TRAIN="${PRUNE_REPLAY_AFTER_TRAIN:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
RUN_BASE_EVALS="${RUN_BASE_EVALS:-1}"
RUN_VARIANTS="${RUN_VARIANTS:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

# XLA can resolve libdevice relative to the process working directory.
cd "$PROJECT_ROOT"

if [[ -d "$PROJECT_ROOT/.deps/craftax_pkgs" ]]; then
  export PYTHONPATH="$PROJECT_ROOT/.deps/craftax_pkgs${PYTHONPATH:+:$PYTHONPATH}"
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

if [[ ! -d "$TRAIN_ROOT/train/ckpt" ]]; then
  echo "Missing checkpoint directory: $TRAIN_ROOT/train/ckpt" >&2
  exit 1
fi

REF_CKPT="$(latest_ckpt "$TRAIN_ROOT/train")"
if [[ -z "$REF_CKPT" ]]; then
  echo "No checkpoint found under $TRAIN_ROOT/train/ckpt" >&2
  exit 1
fi

echo "===================================================="
echo "[Craftax multi-seed fault queue]"
echo "root          : $ROOT"
echo "train root    : $TRAIN_ROOT"
echo "reference ckpt: $REF_CKPT"
echo "seeds         : $SEEDS"
echo "variants      : $VARIANTS"
echo "splits        : $EVAL_SPLITS"
echo "train steps   : $TRAIN_STEPS"
echo "milestones    : ${TRAIN_MILESTONES:-disabled}"
if [[ -n "$TRAIN_MILESTONES" ]]; then
  echo "interim eval  : $MILESTONE_EVAL_STEPS / sparse $MILESTONE_SPARSE_EVAL_STEPS"
  echo "final eval    : $ADAPT_EVAL_STEPS / sparse $SPARSE_EVAL_STEPS"
fi
echo "train ratio   : $TRAIN_RATIO"
echo "replay size   : $REPLAY_SIZE"
echo "min free GB   : $MIN_FREE_GB"
echo "prune replay  : $PRUNE_REPLAY_AFTER_TRAIN"
echo "rnd override  : ${RND_ALPHA:-variant default}"
echo "jax platform  : $JAX_PLATFORM"
echo "python        : $PYTHON_BIN"
echo "PYTHONPATH    : ${PYTHONPATH:-}"
echo "LD_LIBRARY    : ${LD_LIBRARY_PATH:-}"
echo "XLA_FLAGS     : ${XLA_FLAGS:-}"
echo "===================================================="

clean_fault_env() {
  export CRAFTAX_FAULT=0
  export CRAFTAX_FAULT_SAMPLER=0
  unset CRAFTAX_FAULT_PROFILE || true
  unset CRAFTAX_FAULT_EP_PROB || true
  unset CRAFTAX_FAULT_MANIFEST_PROB || true
}

fault_env() {
  local profile="$1"
  local ep_prob="$2"
  local manifest_prob="$3"
  export CRAFTAX_FAULT=0
  export CRAFTAX_FAULT_SAMPLER=1
  export CRAFTAX_FAULT_PROFILE="$profile"
  export CRAFTAX_FAULT_EP_PROB="$ep_prob"
  export CRAFTAX_FAULT_MANIFEST_PROB="$manifest_prob"
}

rnd_env_off() {
  export CRAFTAX_USE_RND=0
  export CRAFTAX_RND_UPDATE=0
}

variant_is_rnd() {
  local variant="$1"
  case "$variant" in
    rnd_beta005|rnd_beta01|rnd_beta02)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

variant_rnd_alpha() {
  local variant="$1"
  if [[ -n "$RND_ALPHA" ]]; then
    echo "$RND_ALPHA"
    return
  fi
  case "$variant" in
    rnd_beta005) echo "0.05" ;;
    rnd_beta01) echo "0.1" ;;
    rnd_beta02) echo "0.2" ;;
    *) echo "0.05" ;;
  esac
}

rnd_env_on() {
  local seed="$1"
  local alpha="$2"
  export CRAFTAX_USE_RND=1
  export CRAFTAX_RND_ALPHA="$alpha"
  export CRAFTAX_RND_UPDATE=1
  export CRAFTAX_RND_NORM="$RND_NORM"
  export CRAFTAX_RND_CLIP="$RND_CLIP"
  export CRAFTAX_RND_DOWNSAMPLE="$RND_DOWNSAMPLE"
  export CRAFTAX_RND_HIDDEN_DIM="$RND_HIDDEN_DIM"
  export CRAFTAX_RND_OUTPUT_DIM="$RND_OUTPUT_DIM"
  export CRAFTAX_RND_LR="$RND_LR"
  export CRAFTAX_RND_SEED="$seed"
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

run_eval() {
  local seed="$1"
  local name="$2"
  local ckpt="$3"
  local steps="$4"
  local profile="$5"
  local ep_prob="$6"
  local manifest_prob="$7"
  local stats="${8:-}"
  local norm_mode="${9:-p95}"
  local score_source="${10:-latent_reward}"
  local use_kl_bound="${11:-False}"
  local suspicious_threshold="${12:-1.0}"
  local outdir="${EVAL_OUTPUT_ROOT:-$ROOT}/seed_${seed}/$name"

  if [[ -n "$profile" ]]; then
    fault_env "$profile" "$ep_prob" "$manifest_prob"
  else
    clean_fault_env
  fi
  export CRAFTAX_FAULT_SEED="$seed"
  rnd_env_off

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
    --env.craftax.platform "" \
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
    --fault.ref_ckpt "$REF_CKPT" \
    --fault.norm_stats "$stats" \
    --fault.norm_mode "$norm_mode" \
    --fault.score_source "$score_source" \
    --fault.use_kl_bound "$use_kl_bound" \
    --fault.suspicious_threshold "$suspicious_threshold" \
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

data = {}
data.update(summarize("latent_kl", values("latent_kl_surprise", "fault/latent_kl_surprise", "log/ref_latent_kl_surprise")))
data.update(summarize("reward_error", values("reward_prediction_error", "fault/reward_prediction_error", "log/ref_reward_prediction_error")))
data.update(summarize("fault_score", values("fault_score_raw", "fault/fault_score_raw", "log/ref_fault_score_raw")))
context_scores = {}
for row, score in zip(rows, values("fault_score_raw", "fault/fault_score_raw", "log/ref_fault_score_raw")):
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
print(json.dumps(data, indent=2, sort_keys=True))
PY
}

select_variant_stats() {
  local seed="$1"
  local variant="$2"
  local default_stats="$3"
  local params extensions norm_mode score_source use_kl_bound suspicious_threshold
  params="$(variant_params "$variant")"
  read -r _ _ _ _ _ _ _ norm_mode _ <<< "$params"
  extensions="$(variant_extensions "$variant")"
  read -r score_source use_kl_bound _ suspicious_threshold _ <<< "$extensions"
  VARIANT_STATS_FILE="$default_stats"

  if [[ "$score_source" == "kl_bound" && "$norm_mode" != "none" ]]; then
    local seed_root="$ROOT/seed_${seed}"
    local clean_name="base_clean_eval_${score_source}"
    VARIANT_STATS_FILE="$seed_root/craftax_clean_fault_stats_${score_source}.json"
    if [[ ! -f "$VARIANT_STATS_FILE" ]]; then
      if [[ ! -f "$seed_root/$clean_name/fault_score_trace.jsonl" ]]; then
        run_job "$seed" "${clean_name}_for_stats" \
          run_eval "$seed" "$clean_name" "$REF_CKPT" "$BASE_EVAL_STEPS" "" "" "" "" "none" "$score_source" "$use_kl_bound" "$suspicious_threshold"
      fi
      run_job "$seed" "make_norm_stats_${score_source}" \
        make_norm_stats "$seed_root/$clean_name/fault_score_trace.jsonl" "$VARIANT_STATS_FILE"
    fi
  fi
}

variant_params() {
  local variant="$1"
  case "$variant" in
    taskonly)
      echo "True dense 0.0 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    dense_beta01)
      echo "False dense 0.1 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    dense_beta02)
      echo "False dense 0.2 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    threshold_p95_beta01)
      echo "False threshold 0.1 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    excess_p95_beta01)
      echo "False excess_threshold 0.1 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    excess_p95_beta02)
      echo "False excess_threshold 0.2 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    delta_p95_beta01)
      echo "False delta_threshold 0.1 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    delta_p95_beta02)
      echo "False delta_threshold 0.2 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    delta_p95_beta04)
      echo "False delta_threshold 0.4 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    excess_delta_p90_beta02)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none p90 0.0 0.0 0.0 False 0.0"
      ;;
    excess_delta_p95_beta01)
      echo "False excess_delta_threshold 0.1 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    excess_delta_p95_beta02)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    excess_delta_p95_beta04)
      echo "False excess_delta_threshold 0.4 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    excess_delta_p99_beta02)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none p99 0.0 0.0 0.0 False 0.0"
      ;;
    bugonly_from_scratch)
      # Standard DreamerV3 trained from scratch in the fault-seeded environment.
      # Fault scoring is disabled during training and enabled only at evaluation
      # time with the frozen clean reference model for comparable diagnostics.
      echo "True dense 0.0 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    rnd_beta005|rnd_beta01|rnd_beta02)
      # DreamerV3 + Random Network Distillation intrinsic reward baseline.
      # Fault score remains logging-only so evaluation traces are comparable.
      echo "True dense 0.0 1.0 0.5 2.0 none p95 0.0 0.0 0.0 False 0.0"
      ;;
    contextual_excess_delta_beta01)
      echo "False excess_delta_threshold 0.1 1.0 0.5 2.0 none context_p95 0.0 0.0 0.0 False 0.0"
      ;;
    contextual_excess_delta_beta02)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none context_p95 0.0 0.0 0.0 False 0.0"
      ;;
    contextual_excess_delta_beta05)
      echo "False excess_delta_threshold 0.5 1.0 0.5 2.0 none context_p95 0.0 0.0 0.0 False 0.0"
      ;;
    contextual_excess_delta_p90_beta02)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none context_p90 0.0 0.0 0.0 False 0.0"
      ;;
    contextual_excess_delta_p99_beta02)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none context_p99 0.0 0.0 0.0 False 0.0"
      ;;
    contextual_crl_task85)
      echo "False excess_delta_threshold 1.0 1.0 0.5 2.0 none context_p95 0.0 0.0 0.0 False 0.0"
      ;;
    contextual_coverage_beta02)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none context_p95 0.01 0.0 0.0 False 0.0"
      ;;
    contextual_coverage_lam002)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none context_p95 0.002 0.0 0.0 False 0.0"
      ;;
    contextual_coverage_lam005)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none context_p95 0.005 0.0 0.0 False 0.0"
      ;;
    contextual_unique_beta02)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none context_p95 0.01 0.05 0.01 False 0.0"
      ;;
    contextual_adaptive_beta02)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none context_p95 0.01 0.05 0.01 True $ADAPTIVE_TASK_TARGET"
      ;;
    klbound_reward_beta02)
      echo "False dense 0.2 0.0 0.0 1.0 none none 0.0 0.0 0.0 False 0.0"
      ;;
    klbound_crl_task85)
      echo "False dense 1.0 0.0 0.0 1.0 none none 0.0 0.0 0.0 False 0.0"
      ;;
    klbound_contextual_beta02)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none context_p95 0.0 0.0 0.0 False 0.0"
      ;;
    klbound_contextual_crl_beta02)
      echo "False excess_delta_threshold 0.2 1.0 0.5 2.0 none context_p95 0.0 0.0 0.0 False 0.0"
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      return 1
      ;;
  esac
}

variant_extensions() {
  local variant="$1"
  case "$variant" in
    klbound_reward_beta02)
      echo "kl_bound True none 0.0 1.0 0.05 0.0 10.0 10"
      ;;
    klbound_crl_task85)
      echo "kl_bound True task_lower_bound 0.0 1.0 0.05 0.0 10.0 $CONSTRAINT_WARMUP_EPISODES"
      ;;
    klbound_contextual_beta02)
      echo "kl_bound True none 1.0 1.0 0.05 0.0 10.0 10"
      ;;
    klbound_contextual_crl_beta02)
      echo "kl_bound True task_lower_bound_scaled 1.0 1.0 0.05 0.0 10.0 $CONSTRAINT_WARMUP_EPISODES"
      ;;
    contextual_crl_task85)
      echo "latent_reward False task_lower_bound 1.0 1.0 0.05 0.0 10.0 $CONSTRAINT_WARMUP_EPISODES"
      ;;
    *)
      echo "latent_reward False none 1.0 1.0 0.05 0.0 10.0 10"
      ;;
  esac
}

run_train_variant() {
  local seed="$1"
  local variant="$2"
  local stats="$3"
  local target_steps="${4:-$TRAIN_STEPS}"
  local outdir="$ROOT/seed_${seed}/train_${variant}/train"

  if [[ "$variant" == "bugonly_from_scratch" ]]; then
    fault_env "benchmark_train" "0.5" "1.0"
    export CRAFTAX_FAULT_SEED="$seed"
    rnd_env_off

    "$PYTHON_BIN" "$MAIN_PY" \
      --script train \
      --configs $CONFIGS \
      --logdir "$outdir" \
      --seed "$seed" \
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
      --fault.enabled False
    return
  fi

  local params
  params="$(variant_params "$variant")"
  read -r log_only reward_mode beta reward_threshold delta_threshold clip reward_gate norm_mode coverage_beta unique_beta repeat_penalty adaptive_beta adaptive_target <<< "$params"
  local extensions score_source use_kl_bound constraint_mode suspicious_threshold
  local constraint_lambda_init constraint_lambda_lr constraint_lambda_min
  local constraint_lambda_max constraint_warmup
  extensions="$(variant_extensions "$variant")"
  read -r score_source use_kl_bound constraint_mode suspicious_threshold constraint_lambda_init constraint_lambda_lr constraint_lambda_min constraint_lambda_max constraint_warmup <<< "$extensions"

  fault_env "benchmark_train" "0.5" "1.0"
  export CRAFTAX_FAULT_SEED="$seed"
  local rnd_alpha
  if variant_is_rnd "$variant"; then
    rnd_alpha="$(variant_rnd_alpha "$variant")"
    rnd_env_on "$seed" "$rnd_alpha"
    echo "[rnd] variant=$variant alpha=$rnd_alpha norm=$RND_NORM clip=$RND_CLIP"
  else
    rnd_env_off
  fi

  "$PYTHON_BIN" "$MAIN_PY" \
    --script train \
    --configs $CONFIGS \
    --logdir "$outdir" \
    --seed "$seed" \
    --run.from_checkpoint "$REF_CKPT" \
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
    --fault.ref_ckpt "$REF_CKPT" \
    --fault.norm_stats "$stats" \
    --fault.log_only "$log_only" \
    --fault.beta "$beta" \
    --fault.reward_mode "$reward_mode" \
    --fault.norm_mode "$norm_mode" \
    --fault.score_source "$score_source" \
    --fault.use_kl_bound "$use_kl_bound" \
    --fault.reward_threshold "$reward_threshold" \
    --fault.reward_delta_threshold "$delta_threshold" \
    --fault.clip "$clip" \
    --fault.reward_gate "$reward_gate" \
    --fault.semantic_coverage_beta "$coverage_beta" \
    --fault.unique_suspicious_beta "$unique_beta" \
    --fault.repeat_suspicious_penalty "$repeat_penalty" \
    --fault.adaptive_beta "$adaptive_beta" \
    --fault.adaptive_task_target "$adaptive_target" \
    --fault.suspicious_threshold "$suspicious_threshold" \
    --fault.constraint_mode "$constraint_mode" \
    --fault.constraint_task_target "$CONSTRAINT_TASK_TARGET" \
    --fault.constraint_lambda_init "$constraint_lambda_init" \
    --fault.constraint_lambda_lr "$constraint_lambda_lr" \
    --fault.constraint_lambda_min "$constraint_lambda_min" \
    --fault.constraint_lambda_max "$constraint_lambda_max" \
    --fault.constraint_warmup_episodes "$constraint_warmup" \
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

run_variant_evals() {
  local seed="$1"
  local variant="$2"
  local ckpt="$3"
  local stats="$4"
  local milestone="${5:-}"
  local params norm_mode
  local extensions score_source use_kl_bound suspicious_threshold
  params="$(variant_params "$variant")"
  read -r _ _ _ _ _ _ _ norm_mode _ <<< "$params"
  extensions="$(variant_extensions "$variant")"
  read -r score_source use_kl_bound _ suspicious_threshold _ <<< "$extensions"
  local split profile steps
  local previous_eval_root="${EVAL_OUTPUT_ROOT:-}"
  if [[ -n "$milestone" ]]; then
    EVAL_OUTPUT_ROOT="$ROOT/milestone_${milestone}"
  fi
  for split in $EVAL_SPLITS; do
    profile="$(split_profile "$split")"
    steps="$(split_steps "$split" "$milestone")"
    local eval_name="eval_${variant}_${split}"
    local job_name="$eval_name"
    if [[ -n "$milestone" ]]; then
      job_name="${eval_name}_at_${milestone}"
    fi
    run_job "$seed" "$job_name" \
      run_eval "$seed" "eval_${variant}_${split}" "$ckpt" "$steps" "$profile" "0.5" "1.0" "$stats" "$norm_mode" "$score_source" "$use_kl_bound" "$suspicious_threshold"
  done
  if [[ -n "$previous_eval_root" ]]; then
    EVAL_OUTPUT_ROOT="$previous_eval_root"
  else
    unset EVAL_OUTPUT_ROOT || true
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

validate_milestones

for seed in $SEEDS; do
  seed_root="$ROOT/seed_${seed}"
  mkdir -p "$seed_root"

  if [[ "$RUN_BASE_EVALS" == "1" ]]; then
    run_job "$seed" "base_clean_eval" \
      run_eval "$seed" "base_clean_eval" "$REF_CKPT" "$BASE_EVAL_STEPS" "" "" ""
    run_job "$seed" "base_seen_eval" \
      run_eval "$seed" "base_seen_eval" "$REF_CKPT" "$BASE_EVAL_STEPS" "benchmark_seen" "0.5" "1.0"
    run_job "$seed" "base_holdout_eval" \
      run_eval "$seed" "base_holdout_eval" "$REF_CKPT" "$BASE_EVAL_STEPS" "benchmark_holdout" "0.5" "1.0"
    run_job "$seed" "base_sparse_eval" \
      run_eval "$seed" "base_sparse_eval" "$REF_CKPT" "$SPARSE_EVAL_STEPS" "benchmark_sparse" "0.1" "1.0"
  fi

  STATS_FILE="$seed_root/craftax_clean_fault_stats.json"
  if [[ ! -f "$STATS_FILE" ]]; then
    if [[ ! -f "$seed_root/base_clean_eval/fault_score_trace.jsonl" ]]; then
      run_job "$seed" "base_clean_eval_for_stats" \
        run_eval "$seed" "base_clean_eval" "$REF_CKPT" "$BASE_EVAL_STEPS" "" "" ""
    fi
    run_job "$seed" "make_norm_stats" \
      make_norm_stats "$seed_root/base_clean_eval/fault_score_trace.jsonl" "$STATS_FILE"
  fi

  if [[ "$RUN_VARIANTS" == "1" ]]; then
    for variant in $VARIANTS; do
      select_variant_stats "$seed" "$variant" "$STATS_FILE"
      if [[ -n "$TRAIN_MILESTONES" ]]; then
        for milestone in $TRAIN_MILESTONES; do
          run_job "$seed" "train_${variant}_to_${milestone}" \
            run_train_variant "$seed" "$variant" "$VARIANT_STATS_FILE" "$milestone"
          variant_ckpt="$(latest_ckpt "$seed_root/train_${variant}/train")"
          if [[ -z "$variant_ckpt" ]]; then
            echo "Missing checkpoint: seed=$seed variant=$variant milestone=$milestone" >&2
            exit 1
          fi
          milestone_ckpt="$(archive_milestone_ckpt "$seed" "$variant" "$milestone" "$variant_ckpt")"
          run_variant_evals "$seed" "$variant" "$milestone_ckpt" "$VARIANT_STATS_FILE" "$milestone"
        done
      else
        run_job "$seed" "train_${variant}" \
          run_train_variant "$seed" "$variant" "$VARIANT_STATS_FILE"
        variant_ckpt="$(latest_ckpt "$seed_root/train_${variant}/train")"
        if [[ -z "$variant_ckpt" ]]; then
          echo "Missing variant checkpoint for seed=$seed variant=$variant" >&2
          exit 1
        fi
        run_variant_evals "$seed" "$variant" "$variant_ckpt" "$VARIANT_STATS_FILE"
      fi
      prune_replay "$seed_root/train_${variant}/train"
    done
  fi
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
    if [[ -f "$ANALYZE_MILESTONES_PY" ]]; then
      run_job "all" "analysis_learning_curve" \
        "$PYTHON_BIN" "$ANALYZE_MILESTONES_PY" \
          --root "$ROOT" --milestones $TRAIN_MILESTONES
    fi
  else
    run_job "all" "analysis" \
      "$PYTHON_BIN" "$ANALYZE_PY" \
        --root "$ROOT" \
        --outdir "$ROOT/analysis" \
        --baseline taskonly --eval-only
  fi
fi

echo "===================================================="
echo "[Craftax multi-seed fault queue] DONE"
echo "root   : $ROOT"
echo "status : $STATUS_FILE"
echo "log    : $LAUNCHER_LOG"
echo "analysis: $ROOT/analysis"
echo "===================================================="
