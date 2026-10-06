#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/sungmin/le-wm"
PYTHON_BIN="/home/sungmin/.conda/envs/lewm/bin/python"
BASE_DIR="/data/sungmin/lewm/runs"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${BASE_DIR}/pusht_4gpu_${RUN_TAG}}"
SESSION_PREFIX="pusht4_${RUN_TAG}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/hydra" "${RUN_ROOT}/results"
ln -sfn "${RUN_ROOT}" "${BASE_DIR}/pusht_4gpu_latest"

launch() {
  local name="$1"
  local command="$2"
  if tmux has-session -t "${name}" 2>/dev/null; then
    echo "tmux session already exists: ${name}" >&2
    exit 1
  fi
  tmux new-session -d -s "${name}" "bash -lc '${command}'"
}

launch "${SESSION_PREFIX}_train_s3072" \
  "cd ${REPO_ROOT} && set -o pipefail && CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -u train.py seed=3072 data.dataset.name=pusht_expert_train.h5 subdir=pusht_lewm_retrain_3072_${RUN_TAG} hydra.run.dir=${RUN_ROOT}/hydra/train_seed3072 2>&1 | tee ${RUN_ROOT}/logs/train_seed3072.log"

launch "${SESSION_PREFIX}_train_s3073" \
  "cd ${REPO_ROOT} && set -o pipefail && CUDA_VISIBLE_DEVICES=1 ${PYTHON_BIN} -u train.py seed=3073 data.dataset.name=pusht_expert_train.h5 subdir=pusht_lewm_retrain_3073_${RUN_TAG} hydra.run.dir=${RUN_ROOT}/hydra/train_seed3073 2>&1 | tee ${RUN_ROOT}/logs/train_seed3073.log"

launch "${SESSION_PREFIX}_eval_s42" \
  "cd ${REPO_ROOT} && set -o pipefail && export CUDA_VISIBLE_DEVICES=2 && ${PYTHON_BIN} -u eval.py scenario=short seed=42 output.root=${RUN_ROOT}/results hydra.run.dir=${RUN_ROOT}/hydra/eval_seed42_short 2>&1 | tee ${RUN_ROOT}/logs/eval_seed42_short.log && ${PYTHON_BIN} -u eval.py scenario=long seed=42 output.root=${RUN_ROOT}/results hydra.run.dir=${RUN_ROOT}/hydra/eval_seed42_long 2>&1 | tee ${RUN_ROOT}/logs/eval_seed42_long.log"

launch "${SESSION_PREFIX}_eval_s43" \
  "cd ${REPO_ROOT} && set -o pipefail && export CUDA_VISIBLE_DEVICES=3 && ${PYTHON_BIN} -u eval.py scenario=short seed=43 output.root=${RUN_ROOT}/results hydra.run.dir=${RUN_ROOT}/hydra/eval_seed43_short 2>&1 | tee ${RUN_ROOT}/logs/eval_seed43_short.log && ${PYTHON_BIN} -u eval.py scenario=long seed=43 output.root=${RUN_ROOT}/results hydra.run.dir=${RUN_ROOT}/hydra/eval_seed43_long 2>&1 | tee ${RUN_ROOT}/logs/eval_seed43_long.log"

printf 'RUN_ROOT=%s\nSESSION_PREFIX=%s\n' "${RUN_ROOT}" "${SESSION_PREFIX}"
tmux list-sessions -F '#{session_name}' | grep "^${SESSION_PREFIX}_"

