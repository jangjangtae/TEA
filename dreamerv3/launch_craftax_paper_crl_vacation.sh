#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/dreamerv3}"
ROOT="${ROOT:-$HOME/logdir/craftax_paper_crl_vacation_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"
cd "$PROJECT_ROOT"

nohup env ROOT="$ROOT" \
  "$PROJECT_ROOT/dreamerv3/run_craftax_paper_crl_vacation.sh" \
  > "$ROOT/launcher.nohup.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$ROOT/launcher.pid"

echo "Craftax vacation queue started."
echo "PID : $pid"
echo "ROOT: $ROOT"
echo "LOG : $ROOT/launcher.nohup.log"
echo "STATUS: $ROOT/phase_status.tsv"
