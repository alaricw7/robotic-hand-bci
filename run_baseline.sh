#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs/baseline results/baseline

NUM_WORKERS="${NUM_WORKERS:-6}"
TORCH_THREADS="${TORCH_THREADS:-4}"
EXP_NAME="${EXP_NAME:-baseline_t7_swa}"
UV_BIN="${UV_BIN:-uv}"
UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-${UV_INDEX_URL}}"
export UV_INDEX_URL UV_DEFAULT_INDEX

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  PY_CMD=("${UV_BIN}" run --locked python)
fi

"${PY_CMD[@]}" -u run_baseline.py \
  --exp-name "${EXP_NAME}" \
  --preset standard_coords \
  --subject all \
  --split-mode pooled \
  --train-size 0.7 \
  --val-size 0.2 \
  --test-size 0.1 \
  --epochs 200 \
  --save-ckpts \
  --torch-threads "${TORCH_THREADS}" \
  --num-workers "${NUM_WORKERS}" \
  --override mixup_enabled=True mixup_alpha=0.1 \
  --tri-override \
    "time_pool_mode='attn'" \
    "aux_loss_enabled=True" \
    "aux_loss_weight=0.3" \
    "per_branch_norm=True" \
    "freq_bands='lowfreq_dense'" \
    "tri_freq_taps=251" \
    "swa_enabled=True" \
    "swa_start_frac=0.75" \
  "$@"
