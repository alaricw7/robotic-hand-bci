"""
model.py - Baseline EEGNet 模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetBaseline(nn.Module):
    def __init__(
        self,
        n_channels=59,
        n_samples=1126,
        n_classes=6,
        n_temporal_filters=16,
        temporal_kernel=125,
        depth_multiplier=2,
        separable_kernel=16,
        dropout=0.25,
    ):
        super().__init__()

        F1 = n_temporal_filters
        D = depth_multiplier
        F2 = F1 * D

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.spatial_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_depthwise = nn.Conv2d(
            in_channels=F2,
            out_channels=F2,
            kernel_size=(1, separable_kernel),
            padding=(0, separable_kernel // 2),
            groups=F2,
            bias=False,
        )
        self.sep_pointwise = nn.Conv2d(
            in_channels=F2,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feat_dim = self._forward_features(dummy).shape[1]
        self.classifier = nn.Linear(feat_dim, n_classes)

    def _forward_features(self, x):
        # x: (B, 1, C, T)
        in_T = x.shape[-1]
        x = self.temporal_conv(x)
        # 偶数 kernel + padding=k//2 会让输出比输入多 1，裁掉末尾保持 same length。
        if x.shape[-1] > in_T:
            x = x[..., :in_T]
        x = self.bn1(x)

        x = self.bn2(self.spatial_conv(x))
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        pre_sep_T = x.shape[-1]
        x = self.sep_depthwise(x)
        if x.shape[-1] > pre_sep_T:
            x = x[..., :pre_sep_T]
        x = self.bn3(self.sep_pointwise(x))
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feat = self._forward_features(x)
        return self.classifier(feat)

    @staticmethod
    def _clip_weight_max_norm_(weight, max_norm):
        if max_norm is None or max_norm <= 0:
            return
        with torch.no_grad():
            flat = weight.view(weight.shape[0], -1)
            norm = flat.norm(dim=1).clamp(min=1e-8)
            scale = norm.clamp(max=max_norm) / norm
            weight.mul_(scale.view(-1, *([1] * (weight.dim() - 1))))

    def apply_max_norm_constraints(self, spatial_max_norm=1.0, classifier_max_norm=0.25):
        self._clip_weight_max_norm_(self.spatial_conv.weight, spatial_max_norm)
        self._clip_weight_max_norm_(self.classifier.weight, classifier_max_norm)


def build_model(cfg, model_name="baseline"):
    if model_name == "baseline":
        return EEGNetBaseline(
            n_channels=cfg.n_channels,
            n_samples=cfg.n_samples,
            n_classes=cfg.n_classes,
            n_temporal_filters=cfg.n_temporal_filters,
            temporal_kernel=cfg.temporal_kernel,
            depth_multiplier=cfg.depth_multiplier,
            separable_kernel=cfg.separable_kernel,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown model: {model_name}")
