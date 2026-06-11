#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/mnt/disk/soeeg/miniconda3/envs/eeg_env/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-results/aug_hpo}"
LOG_DIR="${LOG_DIR:-logs_reliability}"
NUM_WORKERS="${NUM_WORKERS:-6}"
TORCH_THREADS="${TORCH_THREADS:-4}"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}"
GPUS=(1 2 3 4 5)
SHARDS=("S1,S2" "S3,S4" "S5,S6" "S7,S8" "S9,S10")

COMMON_ARGS=(
  --ablation full_std_coords
  --n-folds 10
  --epochs 200
  --save-ckpts
  --torch-threads "${TORCH_THREADS}"
  --num-workers "${NUM_WORKERS}"
  --results-root "${RESULTS_ROOT}"
  --override mixup_enabled=True mixup_alpha=0.1
)

T7_TRI=(
  "time_pool_mode='attn'"
  "aux_loss_enabled=True"
  "aux_loss_weight=0.3"
  "per_branch_norm=True"
  "freq_bands='lowfreq_dense'"
  "tri_freq_taps=251"
  "swa_enabled=True"
  "swa_start_frac=0.75"
)

launch_shard() {
  local gpu="$1"
  local exp="$2"
  local subjects="$3"
  shift 3
  local log="${LOG_DIR}/${exp}.log"
  echo "[start] GPU${gpu} ${exp} subjects=${subjects} -> ${log}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u run_aug.py \
    --exp-name "${exp}" \
    --subject "${subjects}" \
    "${COMMON_ARGS[@]}" \
    --tri-override "${T7_TRI[@]}" "$@" \
    >"${log}" 2>&1 &
  LAUNCHED_PID="$!"
}

run_experiment() {
  local exp="$1"
  shift
  local pids=()
  local parts=()
  local launched=()
  for i in "${!SHARDS[@]}"; do
    local shard_idx=$((i + 1))
    local part="${exp}_shard${shard_idx}"
    parts+=("${part}")
    if [ -f "${RESULTS_ROOT}/${part}/results.json" ]; then
      echo "[skip] ${part} already has results.json" >&2
      continue
    fi
    launch_shard "${GPUS[$i]}" "${part}" "${SHARDS[$i]}" "$@"
    pids+=("${LAUNCHED_PID}")
    launched+=("${part}")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [ "${status}" -ne 0 ]; then
    echo "[error] ${exp} failed; check ${LOG_DIR}/${exp}_shard*.log" >&2
    exit "${status}"
  fi

  if [ "${#launched[@]}" -gt 0 ]; then
    echo "[done] ${exp} shards: ${launched[*]}" >&2
  fi

  "${PYTHON_BIN}" merge_sharded_results.py \
    --results-root "${RESULTS_ROOT}" \
    --exp-name "${exp}" \
    --parts "${parts[@]}"
}

run_experiment S_swa_repro \
  "reliability_fusion_enabled=False" \
  "branch_drop_consistency_enabled=False"

run_experiment F2_reliability_swa \
  "reliability_fusion_enabled=True" \
  "reliability_tau=1.0" \
  "branch_drop_consistency_enabled=False"

run_experiment F3_reliability_branchdrop_swa \
  "reliability_fusion_enabled=True" \
  "reliability_tau=1.0" \
  "branch_drop_consistency_enabled=True" \
  "branch_drop_prob=0.3" \
  "branch_drop_consistency_weight=0.1"

"${PYTHON_BIN}" summarize_results.py \
  --results-root "${RESULTS_ROOT}" \
  --runs S_swa_repro F2_reliability_swa F3_reliability_branchdrop_swa \
  --out "${RESULTS_ROOT}/reliability_suite_summary.md"
