# TriDomain Baseline: T7 + SWA

This repository contains only the TriDomain baseline path needed to run the
T7 + SWA setting:

```text
S_swa = T7 + SWA(start=0.75)
AVG acc = 0.5523 +- 0.1904
```

The T7 setting is:

```text
mixup_enabled=True
mixup_alpha=0.1
time_pool_mode='attn'
aux_loss_enabled=True
aux_loss_weight=0.3
per_branch_norm=True
freq_bands='lowfreq_dense'
tri_freq_taps=251
swa_enabled=True
swa_start_frac=0.75
```

Run the full baseline training:

```bash
./run_baseline.sh
```

Run from the copied checkpoints instead of retraining:

```bash
./run_baseline.sh --resume
```

For a quick smoke test:

```bash
./run_baseline.sh --subject S1 --n-folds 2 --epochs 1 --num-workers 0
```

Run the 10-seed Monte-Carlo baseline:

```bash
./run_baseline_mc.sh
```

The launcher defaults to
`/mnt/disk/soeeg/miniconda3/envs/eeg_env/bin/python`. Set
`PYTHON_BIN=/path/to/python` if your environment uses a different executable.
