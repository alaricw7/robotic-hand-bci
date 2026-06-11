#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs_task56 results/aug_hpo

NUM_WORKERS="${NUM_WORKERS:-6}"
TORCH_THREADS="${TORCH_THREADS:-4}"
EXP_NAME="${EXP_NAME:-abl_S5_S_swa}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/disk/soeeg/miniconda3/envs/eeg_env/bin/python}"

"${PYTHON_BIN}" -u run_aug.py \
  --exp-name "${EXP_NAME}" \
  --ablation full_std_coords \
  --subject all \
  --n-folds 10 \
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
