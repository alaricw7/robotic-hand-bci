# robotic-hand-bci
A multimodal brain-computer interface framework for EEG-EOG based robotic hand control.

## Data layout

This project keeps data out of Git and tracks it with DVC.

```text
assets/
  montages/                # electrode position files shared by preprocessing/representation
configs/
  preprocessing/           # preprocessing configs
data/
  raw/
    EEG/                  # original EEG BDF recordings
    EOG/                  # original EOG BDF recordings
  processed/
    pythondata1/          # main preprocessed NPZ/FIF/SET dataset
src/robotic_hand_bci/
  preprocessing/          # preprocessing implementation
  stages/                 # stage entrypoints
  cli.py                  # unified project CLI
```

Current DVC-tracked datasets:

```bash
data/raw/EEG.dvc
data/raw/EOG.dvc
dvc.yaml:prepare_eegnet_noica
```

Git stores only DVC metadata. The actual files are kept locally under
`data/` and in the local DVC cache until a remote is configured.

## Common Commands

Run code through the uv environment:

```bash
uv run python model/10fold_npz/main.py --data_root data/processed/pythondata1
```

Run stages through the project CLI:

```bash
uv run rhbci run prepare-eegnet-noica
uv run rhbci run detect-marker-and-rest --subjects S1
```

Current stage/config convention:

```text
src/robotic_hand_bci/stages/          # stage entrypoints
configs/preprocessing/*.toml          # stage configs
configs/representation/*.toml         # analysis configs
assets/montages/                      # shared static assets
```

Representation examples:

```bash
uv run rhbci run repr-tsne-rdm
uv run rhbci run repr-multifeature-rsa
```

More detail: `docs/representation.md`

Check data state:

```bash
uv run dvc status
```

Run the pipeline:

```bash
uv run dvc repro prepare_eegnet_noica
uv run dvc repro detect_marker_and_rest
uv run dvc repro repr_tsne_rdm
```

Recreate tracked data from a configured DVC remote:

```bash
uv run dvc pull
```

## Training

The project uses uv. Run commands from the repository root unless a command
explicitly changes directories.

The environment is configured with `torch==2.7.1+cu128`, which matches this
machine's NVIDIA driver and enables CUDA training.

Check the environment:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Smoke test the subject-dependent 10-fold pipeline:

```bash
cd model/10fold_npz
CUDA_VISIBLE_DEVICES=0 uv run python main.py \
  --subject S1 \
  --exp_name smoke_uv_10fold_s1 \
  --n_epochs 1 \
  --n_folds 2 \
  --early_stop_patience 1 \
  --batch_size 64 \
  --no_save_model \
  --torch_threads 4
```

Run a full 10-fold experiment for all subjects:

```bash
cd model/10fold_npz
CUDA_VISIBLE_DEVICES=0 uv run python main.py \
  --subject all \
  --exp_name pythondata1_10fold \
  --save_model \
  --torch_threads 4
```

Smoke test the LOSO pipeline:

```bash
cd model/loso_npz
CUDA_VISIBLE_DEVICES=0 uv run python main.py \
  --subject 1 \
  --exp_name smoke_uv_loso_s1 \
  --n_epochs 1 \
  --early_stop_patience 1 \
  --batch_size 64 \
  --skip_bandpass \
  --no_save_model \
  --torch_threads 4
```

Run a full LOSO experiment:

```bash
cd model/loso_npz
CUDA_VISIBLE_DEVICES=0 uv run python main.py \
  --subject all \
  --exp_name pythondata1_loso \
  --skip_bandpass \
  --save_model \
  --torch_threads 4
```

Experiment outputs are written under each model directory's `experiments/`
folder and are ignored by Git.
