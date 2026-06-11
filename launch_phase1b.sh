#!/usr/bin/env bash
# Phase-1b: PAC re-run after the init/injection fix + 3-seed A/B for sinc & wdyn.
#   nf_pac_fix_s42            #1 PAC (RNG-restore + LayerNorm + zero-init residual + amp_norm on)
#   nf_sinc_s{42,1337,2025}   #2 learnable sinc, 3 seeds
#   nf_wdyn_gru_s{42,1337,2025} #3 window-dynamics GRU, 3 seeds
# Compare against the existing 5-seed baseline (baseline_mc_seed42..46): kappa_excl = 0.5428 +- 0.0073.
# GPUs 1-4 free (0/5 busy). Two waves.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs_phase1b results/aug_hpo

PY=/mnt/disk/soeeg/miniconda3/envs/eeg_env/bin/python
BASE_AUG=( --override mixup_enabled=True mixup_alpha=0.1 )
BASE_TRI=( --tri-override
        "time_pool_mode='attn'" "aux_loss_enabled=True" "aux_loss_weight=0.3"
        "per_branch_norm=True" "freq_bands='lowfreq_dense'" "tri_freq_taps=251"
        "swa_enabled=True" "swa_start_frac=0.75" )

run() {  # gpu exp_name seed [extra tri-override tokens...]
  local gpu="$1"; shift; local exp="$1"; shift; local seed="$1"; shift
  local log="logs_phase1b/${exp}.log"
  echo "[launch] GPU${gpu}  ${exp}  seed=${seed}  (+ $*)  -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u run_aug.py \
      --exp-name "${exp}" --ablation full_std_coords \
      --subject all --n-folds 10 --epochs 200 --seed "${seed}" \
      --save-ckpts --torch-threads 4 --num-workers 6 \
      "${BASE_AUG[@]}" "${BASE_TRI[@]}" "$@" \
    >"${log}" 2>&1 &
}

echo "=== Wave 1 (GPU 1-4) ==="
run 1 nf_pac_fix_s42  42   "freq_pac_enabled=True"
run 2 nf_sinc_s42     42   "freq_learnable_bands=True"
run 3 nf_sinc_s1337   1337 "freq_learnable_bands=True"
run 4 nf_sinc_s2025   2025 "freq_learnable_bands=True"
wait

echo "=== Wave 2 (GPU 1-3) ==="
run 1 nf_wdyn_gru_s42   42   "freq_window_dynamics='gru'"
run 2 nf_wdyn_gru_s1337 1337 "freq_window_dynamics='gru'"
run 3 nf_wdyn_gru_s2025 2025 "freq_window_dynamics='gru'"
wait

echo "=== DONE ==="
echo "Baseline (5-seed): kappa_excl = 0.5428 +- 0.0073"
echo "If nf_pac_fix_s42 is within ~1 sigma of baseline, add seeds 1337/2025 for a 3-seed PAC A/B."
