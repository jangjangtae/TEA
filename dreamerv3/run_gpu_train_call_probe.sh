#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
ROOT="${ROOT:-$HOME/logdir/gpu_train_call_probe_$(date +%Y%m%d_%H%M%S)}"

CRAFTAX_DEPS="${CRAFTAX_DEPS_PATH:-$PROJECT_ROOT/.deps/craftax_pkgs}"
if [[ -d "$CRAFTAX_DEPS" ]]; then
  export PYTHONPATH="$CRAFTAX_DEPS${PYTHONPATH:+:$PYTHONPATH}"
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH="$(printf '%s' "$PYTHONPATH" | sed 's/::*/:/g; s/^://; s/:$//')"
  export PYTHONPATH
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-off}"
cd "$PROJECT_ROOT"
if [[ "${USE_SYSTEM_CUDA_LIBS:-1}" == "1" ]]; then
  SYSTEM_CUDA_LIBS="${SYSTEM_CUDA_LIBS:-/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib/python3/dist-packages/tensorflow}"
  export LD_LIBRARY_PATH="$SYSTEM_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [[ -z "${XLA_FLAGS:-}" && -d /usr/lib/cuda/nvvm/libdevice ]]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"
  fi
  if [[ -f /usr/lib/cuda/nvvm/libdevice/libdevice.10.bc && ! -e "$PROJECT_ROOT/libdevice.10.bc" ]]; then
    ln -s /usr/lib/cuda/nvvm/libdevice/libdevice.10.bc "$PROJECT_ROOT/libdevice.10.bc"
  fi
fi

mkdir -p "$ROOT"
STATUS="$ROOT/status.tsv"
: > "$STATUS"

COMMON_ARGS=(
  --batch_size "${BATCH_SIZE:-2}"
  --batch_length "${BATCH_LENGTH:-8}"
  --report_length "${REPORT_LENGTH:-8}"
  --jax.platform "${JAX_PLATFORM:-cuda}"
  --jax.compute_dtype "${JAX_COMPUTE_DTYPE:-float32}"
  --jax.compilation_cache "${JAX_COMPILATION_CACHE:-False}"
  --jax.autotune "${JAX_AUTOTUNE:-0}"
  --jax.deterministic "${JAX_DETERMINISTIC:-False}"
  --jax.prefetch "${JAX_PREFETCH:-False}"
  --jax.profiler "${JAX_PROFILER:-False}"
  --jax.donate_train "${JAX_DONATE_TRAIN:-False}"
  --jax.expect_devices 0
)

run_case() {
  local name="$1"
  shift
  local log="$ROOT/${name}.log"
  echo "====================================================" | tee -a "$log"
  echo "[$name] START $(date '+%F %T')" | tee -a "$log"
  echo "logdir: $ROOT/$name" | tee -a "$log"
  echo "====================================================" | tee -a "$log"
  set +e
  "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/probe_agent_train_call.py" \
    --logdir "$ROOT/$name" \
    "$@" \
    "${COMMON_ARGS[@]}" \
    >> "$log" 2>&1
  local code=$?
  set -e
  if [[ "$code" -eq 0 ]]; then
    echo -e "$name\tDONE\t$code" | tee -a "$STATUS"
  else
    echo -e "$name\tFAIL\t$code" | tee -a "$STATUS"
  fi
}

echo "Root: $ROOT"
echo "Python: $PYTHON_BIN"
echo "PYTHONPATH: ${PYTHONPATH:-}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"
echo "XLA_FLAGS: ${XLA_FLAGS:-}"
echo "Common args: ${COMMON_ARGS[*]}"

run_case crafter_train_call --configs crafter size1m
run_case craftax_train_call --configs craftax size1m

echo "===================================================="
cat "$STATUS"
echo "Logs under: $ROOT"
