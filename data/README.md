# Data Directory

This directory contains local data files managed by DVC.

```text
data/
  raw/
    EEG/
    EOG/
  processed/
    pythondata1/
```

Preprocessing code and assets live outside `data/`:

```text
configs/preprocessing/eegnet_noica.toml
assets/montages/Standard-10-5-Cap385_witheog.elp
src/robotic_hand_bci/preprocessing/
src/robotic_hand_bci/stages/
```

The data folders themselves are ignored by Git. Track data changes with DVC:

```bash
uv run dvc add data/raw/EEG data/raw/EOG
git add data/raw/EEG.dvc data/raw/EOG.dvc
```

`data/processed/pythondata1` is no longer managed by a standalone `.dvc` file.
It is the output of the `prepare_eegnet_noica` stage in `dvc.yaml`:

```bash
uv run dvc repro prepare_eegnet_noica
```
