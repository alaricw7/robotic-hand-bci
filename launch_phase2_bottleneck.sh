#!/usr/bin/env bash
# Phase-2 #4: bottleneck / Perceiver cross-domain fusion on un-pooled tokens
# (cross_branch_attn_enabled=True, cross_branch_attn_mode='bottleneck',
#  cross_branch_fusion_tokens=8), layered on the exact abl_S5_S_swa (T7+SWA).
#   nf_bottleneck_s{42,1337,2025}
# Compare 3-seed mean +- std vs 10-seed baseline: kappa_excl = 0.5426 +- 0.0066.
# GPUs 2-4 (GPU1 may be running the CP seed-42 job).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs_phase2 results/aug_hpo

PY=/mnt/disk/soeeg/miniconda3/envs/eeg_env/bin/python
BASE_AUG=( --override mixup_enabled=True mixup_alpha=0.1 )
BASE_TRI=( --tri-override
        "time_pool_mode='attn'" "aux_loss_enabled=True" "aux_loss_weight=0.3"
        "per_branch_norm=True" "freq_bands='lowfreq_dense'" "tri_freq_taps=251"
        "swa_enabled=True" "swa_start_frac=0.75" )

run() {  # gpu exp seed [extra...]
  local gpu="$1"; shift; local exp="$1"; shift; local seed="$1"; shift
  local log="logs_phase2/${exp}.log"
  echo "[launch] GPU${gpu}  ${exp}  seed=${seed}  (+ $*)  -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u run_aug.py \
      --exp-name "${exp}" --ablation full_std_coords \
      --subject all --n-folds 10 --epochs 200 --seed "${seed}" \
      --save-ckpts --torch-threads 4 --num-workers 6 \
      "${BASE_AUG[@]}" "${BASE_TRI[@]}" "$@" \
    >"${log}" 2>&1 &
}

CB=( "cross_branch_attn_enabled=True" "cross_branch_attn_mode='bottleneck'" )

echo "=== Phase-2 bottleneck (GPU 2-4) ==="
run 2 nf_bottleneck_s42   42   "${CB[@]}"
run 3 nf_bottleneck_s1337 1337 "${CB[@]}"
run 4 nf_bottleneck_s2025 2025 "${CB[@]}"
wait

echo "=== DONE ==="
echo "Baseline (10-seed): kappa_excl = 0.5426 +- 0.0066"
echo "Results: results/aug_hpo/nf_bottleneck_s* ; logs: logs_phase2/"
