# EEG Representation Analysis

表征分析现在和预处理一样，按 `stage + config` 组织，不再从顶层 `representation/` 目录直接跑脚本。

## 结构

```text
configs/representation/                 # 每个表征 stage 的默认 TOML
src/robotic_hand_bci/stages/           # stage 入口
src/robotic_hand_bci/representation/   # 实现代码
src/robotic_hand_bci/representation/legacy/  # 历史脚本归档
artifacts/analysis/representation/     # 默认输出
```

## 常用 stage

```bash
uv run rhbci run repr-tsne-rdm
uv run rhbci run repr-tsne-rdm-block
uv run rhbci run repr-spatial-topomap
uv run rhbci run repr-multifeature-rsa
uv run rhbci run repr-topography-physionet
uv run rhbci run repr-spatial-filters
uv run rhbci run repr-refix-filter-grid
```

对应的 DVC stage：

```bash
uv run dvc repro repr_tsne_rdm
uv run dvc repro repr_tsne_rdm_block
uv run dvc repro repr_multifeature_rsa
uv run dvc repro repr_topography_physionet
uv run dvc repro repr_spatial_filters
uv run dvc repro repr_refix_filter_grid
```

这些 analysis stage 在 `dvc.yaml` 中默认使用 `cache: false` 输出。
也就是它们由流水线驱动，但结果文件默认只保留在本地工作区，不推入 DVC cache。

查看某个 stage 的参数：

```bash
uv run rhbci run repr-tsne-rdm --help
```

切换配置文件：

```bash
uv run rhbci run repr-tsne-rdm --config configs/representation/tsne_rdm.toml
```

## 默认输入

NoICA NPZ：

```text
data/processed/pythondata1/{S1..S10}/{S}_EEGNet_NoICA_uV.npz
```

representation checkpoint：

```text
model/10fold_npz/experiments/pythondata1_npz_repr/representation/checkpoints/{S}_eegnet_noica.pt
```

montage：

```text
assets/montages/Standard-10-5-Cap385_witheog.elp
```

## 默认输出

```text
artifacts/analysis/representation/tsne_rdm/
artifacts/analysis/representation/tsne_rdm_block/
artifacts/analysis/representation/rsa/
artifacts/analysis/representation/spatial_topomap_pythondata2/
artifacts/analysis/representation/topography_physionet/
```

`repr-spatial-filters` 和 `repr-refix-filter-grid` 目前仍沿用脚本内部的历史输出约定，后续可以继续并到 `artifacts/analysis/representation/`。

## 训练前置

如果 checkpoint 不存在，先运行：

```bash
cd model/10fold_npz
uv run python run_pythondata1_npz.py --subject all --exp_name pythondata1_npz_repr
```

## Legacy

旧版频谱和脑地形图脚本没有继续暴露为顶层入口，统一归档到：

```text
src/robotic_hand_bci/representation/legacy/
```
