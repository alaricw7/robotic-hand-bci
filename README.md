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

Run the full pooled baseline training. Each subject is split 70/20/10, then
all train, validation, and test splits are concatenated across subjects:

```bash
uv run ./run_baseline.sh
```

Run from the copied checkpoints instead of retraining:

```bash
uv run ./run_baseline.sh --resume
```

For a quick smoke test:

```bash
uv run ./run_baseline.sh --subject S1 --epochs 1 --num-workers 0
```

Set `BCI_DATA_ROOT` to the directory containing `S1/`, `S2/`, ...
if the data is not under `~/my-data`. The launchers default to `python`.
Set `PYTHON_BIN=/path/to/python` if your environment uses a different
executable.
