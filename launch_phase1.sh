#!/usr/bin/env bash
# Phase-1 orthogonal-feature experiments, one seed (42), layered on the EXACT
# abl_S5_S_swa (T7 + SWA) baseline. One toggle per run so each gain is clean.
#   nf_pac        #1 PAC / Canolty MVL complex mean-vector head
#   nf_sinc       #2 learnable sinc (center, bandwidth) band boundaries
#   nf_wdyn_gru   #3 window-axis dynamics (GRU) before pooling
# Control = existing results/aug_hpo/abl_S5_S_swa/ (NOT rerun; default-off bit-exact).
# GPUs 1-3 in parallel; CPU dataloader workers via --num-workers.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs_phase1 results/aug_hpo

PY=/mnt/disk/soeeg/miniconda3/envs/eeg_env/bin/python
SEED=42

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

run() {  # gpu exp_name [extra tri-override tokens...]
  local gpu="$1"; shift
  local exp="$1"; shift
  local log="logs_phase1/${exp}.log"
  echo "[launch] GPU${gpu}  ${exp}  (+ $*)  -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u run_aug.py \
      --exp-name "${exp}" --ablation full_std_coords \
      --subject all --n-folds 10 --epochs 200 --seed "${SEED}" \
      --save-ckpts --torch-threads 4 --num-workers 6 \
      "${BASE_AUG[@]}" "${BASE_TRI[@]}" "$@" \
    >"${log}" 2>&1 &
}

echo "=== Phase-1 (GPU 1-3, single seed=${SEED}) ==="
run 1 nf_pac       "freq_pac_enabled=True"
run 2 nf_sinc      "freq_learnable_bands=True"
run 3 nf_wdyn_gru  "freq_window_dynamics='gru'"
wait

echo "=== DONE ==="
echo "Control: results/aug_hpo/abl_S5_S_swa/  (excl S2/S3: acc=0.6237, kappa=0.5484)"
echo "New runs: results/aug_hpo/nf_*/results.json ; logs in logs_phase1/"
