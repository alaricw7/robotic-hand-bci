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
./run_baseline.sh
```

Run from the copied checkpoints instead of retraining:

```bash
./run_baseline.sh --resume
```

For a quick smoke test:

```bash
./run_baseline.sh --subject S1 --epochs 1 --num-workers 0
```

Run the baseline over multiple seeds and write a mean+-std table:

```bash
./run_baseline_t7_swa_multiseed.sh
```

Analyze pooled-test weak subjects and classes after one or more pooled runs:

```bash
uv run --locked python analyze_pooled_subjects.py \
  --runs baseline_t7_swa \
  --out results/baseline/pooled_subject_analysis.md
```

Run a cross-subject generalization check:

```bash
./run_generalization_check.sh
MODE=subject_cv ./run_generalization_check.sh
```

Set `BCI_DATA_ROOT` to the directory containing `S1/`, `S2/`, ...
if the data is not under `~/my-data`. The launchers use `uv run --locked`
by default. The project-level `uv.toml` and `uv.lock` point to the Tsinghua
PyPI mirror; set `UV_INDEX_URL` if another mirror is preferred.
