#!/usr/bin/env bash
# A. Run baseline_t7_swa over multiple seeds with conservative worker settings.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs/baseline_multiseed results/baseline

UV_BIN="${UV_BIN:-uv}"
UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-${UV_INDEX_URL}}"
export UV_INDEX_URL UV_DEFAULT_INDEX

SEEDS="${SEEDS:-42 43 44 45 46}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TORCH_THREADS="${TORCH_THREADS:-4}"
EPOCHS="${EPOCHS:-200}"
EXP_PREFIX="${EXP_PREFIX:-baseline_t7_swa_seed}"
RESULTS_ROOT="${RESULTS_ROOT:-results/baseline}"
SUMMARY_OUT="${SUMMARY_OUT:-${RESULTS_ROOT}/baseline_t7_swa_multiseed.md}"

runs=()
for seed in ${SEEDS}; do
  exp="${EXP_PREFIX}${seed}"
  log="logs/baseline_multiseed/${exp}.log"
  runs+=("${exp}")
  echo "[run] seed=${seed} exp=${exp} log=${log}"
  NUM_WORKERS="${NUM_WORKERS}" \
  TORCH_THREADS="${TORCH_THREADS}" \
  EXP_NAME="${exp}" \
    ./run_baseline.sh \
      --seed "${seed}" \
      --epochs "${EPOCHS}" \
      --results-root "${RESULTS_ROOT}" \
      "$@" \
    >"${log}" 2>&1
done

"${UV_BIN}" run --locked python summarize_results.py \
  --results-root "${RESULTS_ROOT}" \
  --runs "${runs[@]}" \
  --out "${SUMMARY_OUT}"

echo "saved: ${SUMMARY_OUT}"
