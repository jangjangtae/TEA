#!/usr/bin/env bash
set -euo pipefail

STATUS_FILE="${STATUS_FILE:-/home/railab/logdir/craftax_bugonly_totalbudget_20260710_155846/full/status.tsv}"
TARGET_SEED="${TARGET_SEED:-2}"
BUGONLY_PGID="${BUGONLY_PGID:-}"
CHECK_INTERVAL="${CHECK_INTERVAL:-10}"

if [[ -z "$BUGONLY_PGID" ]]; then
  echo "BUGONLY_PGID is required" >&2
  exit 2
fi

targets=(
  "eval_bugonly_from_scratch_clean_at_1000000"
  "eval_bugonly_from_scratch_seen_at_1000000"
  "eval_bugonly_from_scratch_holdout_at_1000000"
  "eval_bugonly_from_scratch_sparse_at_1000000"
)

stamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

all_done() {
  local name
  for name in "${targets[@]}"; do
    grep -Fq $'\t'"seed=${TARGET_SEED}"$'\t'"${name}"$'\tDONE' "$STATUS_FILE" || return 1
  done
  return 0
}

echo "[$(stamp)] watcher started"
echo "status file : $STATUS_FILE"
echo "target seed : $TARGET_SEED"
echo "bugonly pgid: $BUGONLY_PGID"

while true; do
  if all_done; then
    echo "[$(stamp)] seed=${TARGET_SEED} 1M eval complete; stopping bugonly queue pgid=${BUGONLY_PGID}"
    if kill -0 "-${BUGONLY_PGID}" 2>/dev/null; then
      kill -TERM "-${BUGONLY_PGID}" 2>/dev/null || true
      sleep 20
      if kill -0 "-${BUGONLY_PGID}" 2>/dev/null; then
        echo "[$(stamp)] bugonly process group still alive after TERM; leaving it for manual inspection"
        exit 1
      fi
    else
      echo "[$(stamp)] process group already gone"
    fi
    echo "[$(stamp)] watcher done"
    exit 0
  fi
  echo "[$(stamp)] waiting for seed=${TARGET_SEED} 1M eval completion"
  sleep "$CHECK_INTERVAL"
done
