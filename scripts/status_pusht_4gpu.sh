#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/data/sungmin/lewm/runs/pusht_4gpu_latest}"

echo "RUN_ROOT=$(readlink -f "${RUN_ROOT}")"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

echo
echo "tmux sessions"
tmux list-sessions -F '#{session_name} #{session_attached} #{session_windows}' 2>/dev/null \
  | grep '^pusht4_' || true

echo
echo "latest log lines"
for log in "${RUN_ROOT}"/logs/*.log; do
  [ -e "${log}" ] || continue
  echo "--- ${log}"
  tail -n 5 "${log}"
done

echo
echo "result rows"
find -L "${RUN_ROOT}/results" -name results.jsonl -type f -print -exec tail -n 1 {} \; \
  2>/dev/null || true

