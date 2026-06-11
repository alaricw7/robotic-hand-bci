# TriDomain Codex: T7 + SWA

This folder is a focused copy of the original `tri_domain` code path needed to
run the best Task 5 setting:

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

Run the full reproduced training:

```bash
cd /home/wong/eeg/selfmodel/tri_domain_codex
./run_t7_swa.sh
```

Run from the copied checkpoints instead of retraining:

```bash
cd /home/wong/eeg/selfmodel/tri_domain_codex
./run_t7_swa.sh --resume
```

For a quick smoke test:

```bash
cd /home/wong/eeg/selfmodel/tri_domain_codex
./run_t7_swa.sh --subject S1 --n-folds 2 --epochs 1 --num-workers 0
```

The launcher defaults to
`/mnt/disk/soeeg/miniconda3/envs/eeg_env/bin/python`. Set
`PYTHON_BIN=/path/to/python` if your environment uses a different executable.
