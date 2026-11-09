#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
ROOT="${ROOT:-$HOME/logdir/craftax_speed_probe_$(date +%Y%m%d_%H%M%S)}"

# For safe local probes, dependencies can live outside the main venv:
#   CRAFTAX_DEPS_PATH=/tmp/craftax_pkgs ./dreamerv3/run_craftax_speed_probe.sh
if [[ -n "${CRAFTAX_DEPS_PATH:-}" ]]; then
  export PYTHONPATH="$CRAFTAX_DEPS_PATH${PYTHONPATH:+:$PYTHONPATH}"
elif ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import craftax
PY
then
  if [[ -d "$PROJECT_ROOT/.deps/craftax_pkgs" ]]; then
    export PYTHONPATH="$PROJECT_ROOT/.deps/craftax_pkgs${PYTHONPATH:+:$PYTHONPATH}"
  elif [[ -d /tmp/craftax_pkgs ]]; then
    export PYTHONPATH="/tmp/craftax_pkgs${PYTHONPATH:+:$PYTHONPATH}"
  fi
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH="$(printf '%s' "$PYTHONPATH" | sed 's/::*/:/g; s/^://; s/:$//')"
  export PYTHONPATH
fi

export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-off}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
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
LAUNCHER_LOG="$ROOT/launcher.log"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

STEPS="${STEPS:-20000}"
ENVS="${ENVS:-16}"
TRAIN_RATIO="${TRAIN_RATIO:-64}"
REPLAY_SIZE="${REPLAY_SIZE:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
REPORT_LENGTH="${REPORT_LENGTH:-32}"
SAVE_EVERY="${SAVE_EVERY:-999999}"
CONFIGS="${CONFIGS:-craftax size1m}"
RUN_DEBUG="${RUN_DEBUG:-False}"
JAX_PLATFORM="${JAX_PLATFORM:-cuda}"
JAX_PREALLOC="${JAX_PREALLOC:-True}"
JAX_COMPUTE_DTYPE="${JAX_COMPUTE_DTYPE:-float32}"
JAX_COMPILATION_CACHE="${JAX_COMPILATION_CACHE:-False}"
JAX_AUTOTUNE="${JAX_AUTOTUNE:-0}"
JAX_DETERMINISTIC="${JAX_DETERMINISTIC:-False}"
JAX_PREFETCH="${JAX_PREFETCH:-False}"
JAX_PROFILER="${JAX_PROFILER:-False}"
JAX_DONATE_TRAIN="${JAX_DONATE_TRAIN:-False}"
CRAFTAX_ENV_SMOKE="${CRAFTAX_ENV_SMOKE:-0}"
PRINT_JAX_DEVICES="${PRINT_JAX_DEVICES:-0}"
CRAFTAX_PACKAGE_CHECK="${CRAFTAX_PACKAGE_CHECK:-0}"
if [[ -z "${XLA_PYTHON_CLIENT_PREALLOCATE:-}" ]]; then
  if [[ "$JAX_PREALLOC" == "True" ]]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
  else
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
  fi
fi

echo "===================================================="
echo "[Craftax DreamerV3 speed probe]"
echo "root        : $ROOT"
echo "configs     : $CONFIGS"
echo "steps       : $STEPS"
echo "envs        : $ENVS"
echo "train_ratio : $TRAIN_RATIO"
echo "run debug   : $RUN_DEBUG"
echo "jax platform: $JAX_PLATFORM"
echo "jax prealloc: $JAX_PREALLOC"
echo "jax dtype   : $JAX_COMPUTE_DTYPE"
echo "jax cache   : $JAX_COMPILATION_CACHE"
echo "jax autotune: $JAX_AUTOTUNE"
echo "jax determ. : $JAX_DETERMINISTIC"
echo "jax prefetch: $JAX_PREFETCH"
echo "jax profiler: $JAX_PROFILER"
echo "jax donate  : $JAX_DONATE_TRAIN"
echo "xla prealloc: ${XLA_PYTHON_CLIENT_PREALLOCATE:-}"
echo "python      : $PYTHON_BIN"
echo "PYTHONPATH  : ${PYTHONPATH:-}"
echo "LD_LIBRARY  : ${LD_LIBRARY_PATH:-}"
echo "XLA_FLAGS   : ${XLA_FLAGS:-}"
echo "===================================================="

if [[ "$CRAFTAX_PACKAGE_CHECK" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
import craftax
print("craftax:", getattr(craftax, "__version__", "no-version"), craftax.__file__)
PY
fi

if [[ "$PRINT_JAX_DEVICES" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
import jax
print("jax:", jax.__version__, "backend=", jax.default_backend(), "devices=", jax.devices())
PY
fi

if [[ "$CRAFTAX_ENV_SMOKE" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
import jax
from embodied.envs.craftax import Craftax

print("craftax env smoke: start", jax.default_backend(), jax.devices())
env = Craftax("classic_pixels", seed=0)
obs = env.step({"reset": True, "action": 0})
for index in range(8):
  obs = env.step({"reset": False, "action": index % 17})
print(
    "craftax env smoke: ok",
    obs["image"].shape,
    obs["image"].dtype,
    "timestep=", int(obs["log/timestep"]))
PY
fi

"$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/main.py" \
  --script train \
  --configs $CONFIGS \
  --logdir "$ROOT/train" \
  --run.steps "$STEPS" \
  --run.envs "$ENVS" \
  --run.train_ratio "$TRAIN_RATIO" \
  --run.log_every 60 \
  --run.report_every 120 \
  --run.save_every "$SAVE_EVERY" \
  --run.debug "$RUN_DEBUG" \
  --replay.size "$REPLAY_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --batch_length "$BATCH_LENGTH" \
  --report_length "$REPORT_LENGTH" \
  --jax.platform "$JAX_PLATFORM" \
  --jax.prealloc "$JAX_PREALLOC" \
  --jax.compute_dtype "$JAX_COMPUTE_DTYPE" \
  --jax.compilation_cache "$JAX_COMPILATION_CACHE" \
  --jax.autotune "$JAX_AUTOTUNE" \
  --jax.deterministic "$JAX_DETERMINISTIC" \
  --jax.prefetch "$JAX_PREFETCH" \
  --jax.profiler "$JAX_PROFILER" \
  --jax.donate_train "$JAX_DONATE_TRAIN" \
  --jax.expect_devices 0

echo "Saved under: $ROOT"
