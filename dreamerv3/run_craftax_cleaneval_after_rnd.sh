#!/usr/bin/env bash
set -euo pipefail

# Wait for the current Craftax RND baseline service to finish, then run a
# CleanEval-only queue. This evaluates the frozen clean checkpoint directly on
# clean/seen/holdout/sparse splits without any adaptation or bug reward.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
TRAIN_ROOT="${TRAIN_ROOT:-$HOME/logdir/craftax_clean_1m_ratio512_saved_20260625_154751}"
ROOT="${ROOT:-$HOME/logdir/craftax_cleaneval_after_rnd_$(date +%Y%m%d_%H%M%S)}"
WAIT_SERVICE="${WAIT_SERVICE:-craftax-rnd-baseline-full-20260714_130703.service}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_IF_RND_FAILED="${RUN_IF_RND_FAILED:-0}"

mkdir -p "$ROOT"
LAUNCHER_LOG="$ROOT/launcher.log"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

stamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

echo "===================================================="
echo "[Craftax CleanEval after RND]"
echo "root        : $ROOT"
echo "train root  : $TRAIN_ROOT"
echo "wait service: $WAIT_SERVICE"
echo "poll seconds: $POLL_SECONDS"
echo "project     : $PROJECT_ROOT"
echo "===================================================="

if [[ -n "$WAIT_SERVICE" ]]; then
  echo "[$(stamp)] waiting for $WAIT_SERVICE"
  while systemctl --user is-active --quiet "$WAIT_SERVICE"; do
    systemctl --user status "$WAIT_SERVICE" --no-pager | sed -n '1,14p' || true
    sleep "$POLL_SECONDS"
  done

  result="$(systemctl --user show "$WAIT_SERVICE" -p Result --value 2>/dev/null || true)"
  active="$(systemctl --user show "$WAIT_SERVICE" -p ActiveState --value 2>/dev/null || true)"
  echo "[$(stamp)] wait service finished: active=${active:-unknown} result=${result:-unknown}"
  if [[ "$RUN_IF_RND_FAILED" != "1" && -n "$result" && "$result" != "success" ]]; then
    echo "[CleanEval] not starting because waited service result is '$result'." >&2
    exit 1
  fi
fi

cd "$PROJECT_ROOT"

ROOT="$ROOT/cleaneval" \
TRAIN_ROOT="$TRAIN_ROOT" \
SEEDS="${SEEDS:-0 1 2}" \
VARIANTS="" \
EVAL_SPLITS="${EVAL_SPLITS:-clean seen holdout sparse}" \
RUN_BASE_EVALS=1 \
RUN_VARIANTS=0 \
RUN_ANALYSIS=0 \
TRAIN_STEPS=1 \
TRAIN_MILESTONES="" \
BASE_EVAL_STEPS="${BASE_EVAL_STEPS:-30000}" \
SPARSE_EVAL_STEPS="${SPARSE_EVAL_STEPS:-60000}" \
EVAL_ENVS="${EVAL_ENVS:-1}" \
TRAIN_ENVS="${TRAIN_ENVS:-16}" \
TRAIN_RATIO="${TRAIN_RATIO:-128}" \
REPLAY_SIZE="${REPLAY_SIZE:-1000}" \
MIN_FREE_GB="${MIN_FREE_GB:-35}" \
STOP_ON_FAIL="${STOP_ON_FAIL:-1}" \
PRUNE_REPLAY_AFTER_TRAIN=1 \
"$PROJECT_ROOT/dreamerv3/run_craftax_multiseed_fault_queue.sh"

echo "===================================================="
echo "[Craftax CleanEval after RND] DONE"
echo "root: $ROOT"
echo "===================================================="
