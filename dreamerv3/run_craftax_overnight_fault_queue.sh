#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
TRAIN_ROOT="${TRAIN_ROOT:?Set TRAIN_ROOT to the Craftax clean run root.}"
ROOT="${ROOT:-$HOME/logdir/craftax_overnight_fault_$(date +%Y%m%d_%H%M%S)}"
WAIT_PID="${WAIT_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"

CONFIGS="${CONFIGS:-craftax size1m}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
TRAIN_ENVS="${TRAIN_ENVS:-16}"
TRAIN_RATIO="${TRAIN_RATIO:-128}"
REPLAY_SIZE="${REPLAY_SIZE:-100000}"
SAVE_EVERY="${SAVE_EVERY:-600}"

BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-50000}"
ADAPT_EVAL_STEPS="${ADAPT_EVAL_STEPS:-30000}"
DIAGNOSTIC_STEPS="${DIAGNOSTIC_STEPS:-20000}"
EVAL_ENVS="${EVAL_ENVS:-1}"

BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
REPORT_LENGTH="${REPORT_LENGTH:-32}"
JAX_PLATFORM="${JAX_PLATFORM:-cuda}"
JAX_PREALLOC="${JAX_PREALLOC:-True}"
JAX_COMPUTE_DTYPE="${JAX_COMPUTE_DTYPE:-float32}"
SMOKE="${SMOKE:-0}"

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
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

echo "===================================================="
echo "[Craftax overnight fault queue]"
echo "root          : $ROOT"
echo "train root    : $TRAIN_ROOT"
echo "wait pid      : ${WAIT_PID:-none}"
echo "train steps   : $TRAIN_STEPS"
echo "train ratio   : $TRAIN_RATIO"
echo "replay size   : $REPLAY_SIZE"
echo "base eval     : $BASE_EVAL_STEPS"
echo "adapt eval    : $ADAPT_EVAL_STEPS"
echo "diag eval     : $DIAGNOSTIC_STEPS"
echo "jax platform  : $JAX_PLATFORM"
echo "smoke mode    : $SMOKE"
echo "python        : $PYTHON_BIN"
echo "===================================================="

if [[ -n "$WAIT_PID" ]]; then
  echo "[queue] $(date '+%F %T') waiting for PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  echo "[queue] $(date '+%F %T') PID $WAIT_PID finished"
fi

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
echo "[queue] reference checkpoint: $REF_CKPT"

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

run_eval() {
  local name="$1"
  local ckpt="$2"
  local steps="$3"
  local profile="$4"
  local ep_prob="$5"
  local manifest_prob="$6"
  local outdir="$ROOT/$name"

  echo "===================================================="
  echo "[$name] $(date '+%F %T') EVAL START"
  echo "ckpt   : $ckpt"
  echo "steps  : $steps"
  echo "profile: ${profile:-clean}"
  echo "===================================================="

  if [[ -n "$profile" ]]; then
    fault_env "$profile" "$ep_prob" "$manifest_prob"
  else
    clean_fault_env
  fi

  "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/main.py" \
    --script eval_only \
    --configs $CONFIGS \
    --logdir "$outdir" \
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
    --fault.log_only True \
    --fault.trace fault_score_trace.jsonl

  echo "[$name] $(date '+%F %T') EVAL DONE"
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

def values(key):
  return np.asarray([float(r.get(key, 0.0)) for r in rows], np.float64)

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
  }

data = {}
data.update(summarize("latent_kl", values("latent_kl_surprise")))
data.update(summarize("reward_error", values("reward_prediction_error")))
data.update(summarize("fault_score", values("fault_score_raw")))
data["steps"] = int(len(rows))
out.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
print("Wrote norm stats:", out)
print(json.dumps(data, indent=2, sort_keys=True))
PY
}

run_train() {
  local name="$1"
  local log_only="$2"
  local reward_mode="$3"
  local beta="$4"
  local threshold="$5"
  local outdir="$ROOT/$name/train"

  echo "===================================================="
  echo "[$name] $(date '+%F %T') TRAIN START"
  echo "log_only   : $log_only"
  echo "reward_mode: $reward_mode"
  echo "beta       : $beta"
  echo "threshold  : $threshold"
  echo "===================================================="

  fault_env "benchmark_train" "0.5" "1.0"

  "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/main.py" \
    --script train \
    --configs $CONFIGS \
    --logdir "$outdir" \
    --run.from_checkpoint "$REF_CKPT" \
    --run.steps "$TRAIN_STEPS" \
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
    --fault.norm_stats "$STATS_FILE" \
    --fault.log_only "$log_only" \
    --fault.beta "$beta" \
    --fault.reward_mode "$reward_mode" \
    --fault.norm_mode p95 \
    --fault.reward_threshold "$threshold" \
    --fault.reward_delta_threshold 0.5 \
    --fault.clip 2.0 \
    --fault.reward_gate none \
    --fault.use_reward_error True \
    --fault.w_reward 0.0 \
    --fault.trace fault_score_trace.jsonl

  echo "[$name] $(date '+%F %T') TRAIN DONE"
}

summarize_all() {
  "$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for trace in sorted(root.glob("**/bug_trace.jsonl")):
  rel = str(trace.parent.relative_to(root))
  rows = []
  with trace.open() as f:
    for line in f:
      if line.strip():
        rows.append(json.loads(line))
  if not rows:
    summary[rel] = {"steps": 0}
    continue
  faults = [r for r in rows if r.get("log/fault_applied", 0)]
  triggers = [r for r in rows if r.get("log/fault_trigger_context", 0)]
  sem = [r for r in rows if r.get("log/semantic_fault_applied", 0)]
  rewards = [float(r.get("reward", 0.0)) for r in rows]
  summary[rel] = {
      "steps": len(rows),
      "fault_applied": len(faults),
      "fault_trigger_context": len(triggers),
      "semantic_fault_applied": len(sem),
      "fault_rate": len(faults) / max(1, len(rows)),
      "trigger_rate": len(triggers) / max(1, len(rows)),
      "reward_mean": sum(rewards) / max(1, len(rewards)),
  }
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print("Wrote", out)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

run_eval "00_base_clean_eval" "$REF_CKPT" "$BASE_EVAL_STEPS" "" "" ""
run_eval "01_base_diagnostic_fault_eval" "$REF_CKPT" "$DIAGNOSTIC_STEPS" "diagnostic" "1.0" "1.0"
if [[ "$SMOKE" == "1" ]]; then
  STATS_FILE="$ROOT/craftax_clean_fault_stats.json"
  make_norm_stats "$ROOT/00_base_clean_eval/fault_score_trace.jsonl" "$STATS_FILE"
  run_train "10_taskonly_faultlog" True dense 0.0 1.0
  TASKONLY_CKPT="$(latest_ckpt "$ROOT/10_taskonly_faultlog/train")"
  run_eval "11_taskonly_seen_eval" "$TASKONLY_CKPT" "$ADAPT_EVAL_STEPS" "benchmark_seen" "0.5" "1.0"
  summarize_all
  echo "===================================================="
  echo "[Craftax overnight fault queue] SMOKE DONE"
  echo "root   : $ROOT"
  echo "stats  : $STATS_FILE"
  echo "summary: $ROOT/summary.json"
  echo "===================================================="
  exit 0
fi
run_eval "02_base_seen_fault_eval" "$REF_CKPT" "$BASE_EVAL_STEPS" "benchmark_seen" "0.5" "1.0"
run_eval "03_base_holdout_fault_eval" "$REF_CKPT" "$BASE_EVAL_STEPS" "benchmark_holdout" "0.5" "1.0"

STATS_FILE="$ROOT/craftax_clean_fault_stats.json"
make_norm_stats "$ROOT/00_base_clean_eval/fault_score_trace.jsonl" "$STATS_FILE"

run_train "10_taskonly_faultlog" True dense 0.0 1.0
TASKONLY_CKPT="$(latest_ckpt "$ROOT/10_taskonly_faultlog/train")"
run_eval "11_taskonly_seen_eval" "$TASKONLY_CKPT" "$ADAPT_EVAL_STEPS" "benchmark_seen" "0.5" "1.0"
run_eval "12_taskonly_holdout_eval" "$TASKONLY_CKPT" "$ADAPT_EVAL_STEPS" "benchmark_holdout" "0.5" "1.0"

run_train "20_fault_dense_beta01" False dense 0.1 1.0
DENSE_CKPT="$(latest_ckpt "$ROOT/20_fault_dense_beta01/train")"
run_eval "21_dense_seen_eval" "$DENSE_CKPT" "$ADAPT_EVAL_STEPS" "benchmark_seen" "0.5" "1.0"
run_eval "22_dense_holdout_eval" "$DENSE_CKPT" "$ADAPT_EVAL_STEPS" "benchmark_holdout" "0.5" "1.0"

run_train "30_fault_excess_threshold_beta01" False excess_threshold 0.1 1.0
THRESH_CKPT="$(latest_ckpt "$ROOT/30_fault_excess_threshold_beta01/train")"
run_eval "31_threshold_seen_eval" "$THRESH_CKPT" "$ADAPT_EVAL_STEPS" "benchmark_seen" "0.5" "1.0"
run_eval "32_threshold_holdout_eval" "$THRESH_CKPT" "$ADAPT_EVAL_STEPS" "benchmark_holdout" "0.5" "1.0"

run_eval "90_base_sparse_eval" "$REF_CKPT" "$ADAPT_EVAL_STEPS" "benchmark_sparse" "0.1" "1.0"
run_eval "91_dense_sparse_eval" "$DENSE_CKPT" "$ADAPT_EVAL_STEPS" "benchmark_sparse" "0.1" "1.0"
run_eval "92_threshold_sparse_eval" "$THRESH_CKPT" "$ADAPT_EVAL_STEPS" "benchmark_sparse" "0.1" "1.0"

summarize_all

echo "===================================================="
echo "[Craftax overnight fault queue] DONE"
echo "root   : $ROOT"
echo "stats  : $STATS_FILE"
echo "summary: $ROOT/summary.json"
echo "===================================================="
