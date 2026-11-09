#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
TRAIN_ROOT="${TRAIN_ROOT:?Set TRAIN_ROOT to the completed Craftax clean run root.}"
ROOT="${ROOT:-$HOME/logdir/craftax_postclean_eval_$(date +%Y%m%d_%H%M%S)}"
WAIT_PID="${WAIT_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"

EVAL_STEPS="${EVAL_STEPS:-50000}"
DIAGNOSTIC_STEPS="${DIAGNOSTIC_STEPS:-20000}"
SPARSE_STEPS="${SPARSE_STEPS:-50000}"
ENVS="${ENVS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
REPORT_LENGTH="${REPORT_LENGTH:-32}"
JAX_PLATFORM="${JAX_PLATFORM:-cuda}"
JAX_PREALLOC="${JAX_PREALLOC:-True}"
JAX_COMPUTE_DTYPE="${JAX_COMPUTE_DTYPE:-float32}"
CONFIGS="${CONFIGS:-craftax size1m}"

if [[ -d "$PROJECT_ROOT/.deps/craftax_pkgs" ]]; then
  export PYTHONPATH="$PROJECT_ROOT/.deps/craftax_pkgs${PYTHONPATH:+:$PYTHONPATH}"
fi

export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-off}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
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
echo "[Craftax post-clean eval queue]"
echo "root       : $ROOT"
echo "train root : $TRAIN_ROOT"
echo "wait pid   : ${WAIT_PID:-none}"
echo "eval steps : $EVAL_STEPS"
echo "diag steps : $DIAGNOSTIC_STEPS"
echo "sparse     : $SPARSE_STEPS"
echo "envs       : $ENVS"
echo "python     : $PYTHON_BIN"
echo "PYTHONPATH : ${PYTHONPATH:-}"
echo "===================================================="

if [[ -n "$WAIT_PID" ]]; then
  echo "[queue] $(date '+%F %T') waiting for PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  echo "[queue] $(date '+%F %T') PID $WAIT_PID finished"
fi

if [[ ! -d "$TRAIN_ROOT/train/ckpt" ]]; then
  echo "Missing checkpoint directory: $TRAIN_ROOT/train/ckpt" >&2
  exit 1
fi

CKPT="$(find "$TRAIN_ROOT/train/ckpt" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [[ -z "$CKPT" ]]; then
  echo "No checkpoint found under $TRAIN_ROOT/train/ckpt" >&2
  exit 1
fi
echo "[queue] using checkpoint: $CKPT"

run_eval() {
  local name="$1"
  local steps="$2"
  local profile="$3"
  local ep_prob="$4"
  local manifest_prob="$5"
  local logdir="$ROOT/$name"

  echo "===================================================="
  echo "[$name] $(date '+%F %T') START"
  echo "logdir       : $logdir"
  echo "steps        : $steps"
  echo "profile      : ${profile:-clean}"
  echo "ep prob      : ${ep_prob:-none}"
  echo "manifest prob: ${manifest_prob:-none}"
  echo "===================================================="

  if [[ -n "$profile" ]]; then
    export CRAFTAX_FAULT_SAMPLER=1
    export CRAFTAX_FAULT=0
    export CRAFTAX_FAULT_PROFILE="$profile"
    export CRAFTAX_FAULT_EP_PROB="$ep_prob"
    export CRAFTAX_FAULT_MANIFEST_PROB="$manifest_prob"
  else
    export CRAFTAX_FAULT_SAMPLER=0
    export CRAFTAX_FAULT=0
    unset CRAFTAX_FAULT_PROFILE || true
    unset CRAFTAX_FAULT_EP_PROB || true
    unset CRAFTAX_FAULT_MANIFEST_PROB || true
  fi

  "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/main.py" \
    --script eval_only \
    --configs $CONFIGS \
    --logdir "$logdir" \
    --run.from_checkpoint "$CKPT" \
    --run.steps "$steps" \
    --run.envs "$ENVS" \
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
    --fault.ref_ckpt "$CKPT" \
    --fault.log_only True \
    --fault.trace fault_score_trace.jsonl

  echo "[$name] $(date '+%F %T') DONE"
}

run_eval "01_clean_eval" "$EVAL_STEPS" "" "" ""
run_eval "02_diagnostic_fault_eval" "$DIAGNOSTIC_STEPS" "diagnostic" "1.0" "1.0"
run_eval "03_seen_fault_eval" "$EVAL_STEPS" "benchmark_seen" "0.5" "1.0"
run_eval "04_holdout_fault_eval" "$EVAL_STEPS" "benchmark_holdout" "0.5" "1.0"
run_eval "05_sparse_fault_eval" "$SPARSE_STEPS" "benchmark_sparse" "0.1" "1.0"

"$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for trace in sorted(root.glob("*/bug_trace.jsonl")):
  name = trace.parent.name
  rows = []
  with trace.open() as f:
    for line in f:
      if line.strip():
        rows.append(json.loads(line))
  if not rows:
    summary[name] = {"steps": 0}
    continue
  faults = [r for r in rows if r.get("log/fault_applied", 0)]
  triggers = [r for r in rows if r.get("log/fault_trigger_context", 0)]
  sem = [r for r in rows if r.get("log/semantic_fault_applied", 0)]
  scores = [r.get("log/score", 0.0) for r in rows if "log/score" in r]
  summary[name] = {
      "steps": len(rows),
      "fault_applied": len(faults),
      "fault_trigger_context": len(triggers),
      "semantic_fault_applied": len(sem),
      "fault_rate": len(faults) / max(1, len(rows)),
      "trigger_rate": len(triggers) / max(1, len(rows)),
      "mean_log_score": sum(scores) / max(1, len(scores)),
  }
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print("Wrote", out)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "===================================================="
echo "[Craftax post-clean eval queue] DONE"
echo "summary: $ROOT/summary.json"
echo "===================================================="
