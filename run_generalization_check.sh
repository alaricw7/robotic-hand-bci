#!/usr/bin/env bash
# C. Run a cross-subject generalization check: LOSO by default, or subject_cv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs/generalization results/baseline

UV_BIN="${UV_BIN:-uv}"
UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-${UV_INDEX_URL}}"
export UV_INDEX_URL UV_DEFAULT_INDEX

MODE="${MODE:-loso}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TORCH_THREADS="${TORCH_THREADS:-4}"
EPOCHS="${EPOCHS:-200}"
N_FOLDS="${N_FOLDS:-10}"
EXP_NAME="${EXP_NAME:-baseline_t7_swa_${MODE}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/baseline}"
LOG_PATH="${LOG_PATH:-logs/generalization/${EXP_NAME}.log}"

case "${MODE}" in
  loso|subject_cv)
    ;;
  *)
    echo "MODE must be loso or subject_cv, got: ${MODE}" >&2
    exit 2
    ;;
esac

"${UV_BIN}" run --locked python -u run_baseline.py \
  --exp-name "${EXP_NAME}" \
  --preset standard_coords \
  --subject all \
  --split-mode "${MODE}" \
  --train-size 0.7 \
  --val-size 0.2 \
  --test-size 0.1 \
  --n-folds "${N_FOLDS}" \
  --epochs "${EPOCHS}" \
  --save-ckpts \
  --torch-threads "${TORCH_THREADS}" \
  --num-workers "${NUM_WORKERS}" \
  --results-root "${RESULTS_ROOT}" \
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
  "$@" \
  >"${LOG_PATH}" 2>&1

echo "saved log: ${LOG_PATH}"
echo "saved results: ${RESULTS_ROOT}/${EXP_NAME}/results.json"
