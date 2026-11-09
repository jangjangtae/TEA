#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/dreamer_cuda/bin/python}"
ROOT="${ROOT:-/tmp/minigrid_smoke_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

echo "MiniGrid smoke root: $ROOT"
"$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/test_minigrid_faults.py"

for config in minigrid minigrid_vision; do
  echo "===================================================="
  echo "Agent train-call smoke: $config"
  "$PYTHON_BIN" "$PROJECT_ROOT/dreamerv3/probe_agent_train_call.py" \
    --configs "$config" debug \
    --logdir "$ROOT/${config}_train_call" \
    --batch_size 2 --batch_length 8 --report_length 8 \
    --jax.platform cpu --jax.prealloc False --jax.expect_devices 0
done

echo "MiniGrid smoke DONE: $ROOT"
