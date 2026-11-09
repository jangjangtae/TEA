#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
ROOT="${ROOT:-$HOME/logdir/minigrid_speed_probe_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-minigrid}"
STEPS="${STEPS:-20000}"
ENVS="${ENVS:-16}"
TRAIN_RATIO="${TRAIN_RATIO:-128}"
REPLAY_SIZE="${REPLAY_SIZE:-20000}"

export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib/python3/dist-packages/tensorflow${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -z "${XLA_FLAGS:-}" && -d /usr/lib/cuda/nvvm/libdevice ]]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"
fi

mkdir -p "$ROOT"
echo "===================================================="
echo "[MiniGrid DreamerV3 speed probe]"
echo "root        : $ROOT"
echo "config      : $CONFIG"
echo "steps       : $STEPS"
echo "envs        : $ENVS"
echo "train ratio : $TRAIN_RATIO"
echo "===================================================="

start="$(date +%s)"
"$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/main.py" \
  --script train --configs "$CONFIG" \
  --logdir "$ROOT/train" \
  --run.steps "$STEPS" --run.envs "$ENVS" \
  --run.train_ratio "$TRAIN_RATIO" \
  --run.log_every 30 --run.report_every 60 --run.save_every 999999 \
  --replay.size "$REPLAY_SIZE" \
  --batch_size 16 --batch_length 64 --report_length 32 \
  --jax.platform cuda --jax.prealloc True --jax.compute_dtype float32 \
  --jax.compilation_cache False --jax.autotune 0 \
  --jax.deterministic False --jax.prefetch False \
  --jax.profiler False --jax.donate_train False --jax.expect_devices 0
end="$(date +%s)"

echo "Elapsed seconds: $((end - start))"
echo "MiniGrid speed probe DONE: $ROOT"
