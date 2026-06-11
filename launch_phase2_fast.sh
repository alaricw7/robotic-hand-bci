#!/usr/bin/env bash
# Fast Phase-2 runner: subject-sharded + GPU-packed job pool + auto-merge.
#
# Why fast: each (exp,seed) is split into 5 subject shards (2 subjects each);
# all shards across all (exp,seed) form one global job pool spread over every
# GPU with PACK jobs per GPU. 144 CPU cores easily feed this. A full 10-subject
# run that took ~2h on one GPU now finishes in ~20-25min. Single seed (42):
# CP + bottleneck = 2 runs x 5 shards = 10 shard jobs, done in ~1 wave (~50min).
#
# Resume-friendly: a shard with results.json is skipped; an (exp,seed) whose
# merged results.json exists is skipped entirely.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY=/mnt/disk/soeeg/miniconda3/envs/eeg_env/bin/python
RESULTS_ROOT=results/aug_hpo
LOG_DIR=logs_phase2
mkdir -p "$LOG_DIR" "$RESULTS_ROOT"

GPUS=(1 2 3 4 5)       # GPU0 reserved for another user
PACK=2                 # concurrent jobs per GPU (each ~55% util => 2 fits)
NUM_WORKERS=4          # dataloader workers per job (12 jobs x 4 = 48 << 144)
TORCH_THREADS=4
SEEDS=(42)
SHARDS=("S1,S2" "S3,S4" "S5,S6" "S7,S8" "S9,S10")
MAXJOBS=$(( ${#GPUS[@]} * PACK ))

BASE_AUG=( --override mixup_enabled=True mixup_alpha=0.1 )
BASE_TRI=( "time_pool_mode='attn'" "aux_loss_enabled=True" "aux_loss_weight=0.3"
           "per_branch_norm=True" "freq_bands='lowfreq_dense'" "tri_freq_taps=251"
           "swa_enabled=True" "swa_start_frac=0.75" )

# experiment name -> extra tri-override tokens
exp_extra() {
  case "$1" in
    cp)         echo "fusion_head_mode='lowrank_cp'" ;;
    bottleneck) echo "cross_branch_attn_enabled=True cross_branch_attn_mode='bottleneck'" ;;
    *) echo "" ;;
  esac
}
EXPERIMENTS=(cp bottleneck)

job_count=0
launch_shard() {  # exp seed shard_idx subjects
  local exp="$1" seed="$2" sidx="$3" subjects="$4"
  local part="nf_${exp}_s${seed}_shard${sidx}"
  if [ -f "${RESULTS_ROOT}/${part}/results.json" ]; then
    echo "[skip] ${part} done"; return; fi
  # throttle to MAXJOBS concurrent
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
  local gpu="${GPUS[$(( job_count % ${#GPUS[@]} ))]}"
  job_count=$(( job_count + 1 ))
  local extra; extra=$(exp_extra "$exp")
  echo "[start] GPU${gpu} ${part} subjects=${subjects}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u run_aug.py \
      --exp-name "${part}" --ablation full_std_coords \
      --subject "${subjects}" --n-folds 10 --epochs 200 --seed "${seed}" \
      --save-ckpts --torch-threads "${TORCH_THREADS}" --num-workers "${NUM_WORKERS}" \
      "${BASE_AUG[@]}" --tri-override "${BASE_TRI[@]}" ${extra} \
    >"${LOG_DIR}/${part}.log" 2>&1 &
}

echo "=== launching pool: ${#EXPERIMENTS[@]} exp x ${#SEEDS[@]} seeds x ${#SHARDS[@]} shards, ${MAXJOBS} concurrent ==="
for exp in "${EXPERIMENTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    [ -f "${RESULTS_ROOT}/nf_${exp}_s${seed}/results.json" ] && { echo "[skip] nf_${exp}_s${seed} merged"; continue; }
    for i in "${!SHARDS[@]}"; do
      launch_shard "$exp" "$seed" "$(( i + 1 ))" "${SHARDS[$i]}"
    done
  done
done
echo "=== all shards launched; waiting ==="
wait

echo "=== merging shards ==="
for exp in "${EXPERIMENTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    final="nf_${exp}_s${seed}"
    [ -f "${RESULTS_ROOT}/${final}/results.json" ] && continue
    parts=(); ok=1
    for i in "${!SHARDS[@]}"; do
      p="nf_${exp}_s${seed}_shard$(( i + 1 ))"
      [ -f "${RESULTS_ROOT}/${p}/results.json" ] || { echo "[warn] missing ${p}, skip merge ${final}"; ok=0; break; }
      parts+=("$p")
    done
    [ "$ok" = 1 ] && "${PY}" merge_sharded_results.py --results-root "${RESULTS_ROOT}" \
        --exp-name "${final}" --parts "${parts[@]}" && echo "[merged] ${final}"
  done
done
echo "=== DONE. Compare nf_{cp,bottleneck}_s{42,1337,2025} vs baseline kappa_excl 0.5426 +- 0.0066 ==="
