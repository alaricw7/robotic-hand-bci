#!/usr/bin/env bash
# Monte-Carlo cross-validation of the abl_S5_S_swa (T7 + SWA) baseline.
# Runs the SAME baseline config under 10 different CV seeds (42..51), each seed
# producing an independent stratified 10-fold random partition. The 10-seed
# mean+-std is the robust baseline reference every candidate combo must beat.
# 10 seeds spread over GPUs 1-5; each GPU runs 2 seeds sequentially (two waves).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs_mc_baseline results/aug_hpo

PY=/mnt/disk/soeeg/miniconda3/envs/eeg_env/bin/python

# Exact T7 + SWA baseline (matches launch_phase1.sh, no nf_* toggle).
BASE_AUG=( --override mixup_enabled=True mixup_alpha=0.1 )
BASE_TRI=( --tri-override
        "time_pool_mode='attn'"
        "aux_loss_enabled=True"
        "aux_loss_weight=0.3"
        "per_branch_norm=True"
        "freq_bands='lowfreq_dense'"
        "tri_freq_taps=251"
        "swa_enabled=True"
        "swa_start_frac=0.75" )

run() {  # gpu seed
  local gpu="$1"; local seed="$2"
  local exp="baseline_mc_seed${seed}"
  local log="logs_mc_baseline/${exp}.log"
  echo "[launch] GPU${gpu}  ${exp}  -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u run_aug.py \
      --exp-name "${exp}" --ablation full_std_coords \
      --subject all --n-folds 10 --epochs 200 --seed "${seed}" \
      --save-ckpts --torch-threads 4 --num-workers 6 \
      "${BASE_AUG[@]}" "${BASE_TRI[@]}" \
    >"${log}" 2>&1
}

# Each GPU runs its two seeds sequentially in the background.
gpu_worker() {  # gpu seedA seedB
  run "$1" "$2"
  run "$1" "$3"
}

echo "=== Monte-Carlo baseline: 10 seeds (42..51) over GPU 1-5 ==="
gpu_worker 1 42 47 &
gpu_worker 2 43 48 &
gpu_worker 3 44 49 &
gpu_worker 4 45 50 &
gpu_worker 5 46 51 &
wait

echo "=== ALL SEEDS DONE ==="
echo "Per-seed results: results/aug_hpo/baseline_mc_seed{42..51}/results.json"
echo "Summarize all seeds into one table:"
echo "  ${PY} summarize_results.py --runs $(printf 'baseline_mc_seed%s ' 42 43 44 45 46 47 48 49 50 51)"
