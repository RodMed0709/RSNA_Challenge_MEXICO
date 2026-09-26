from contextlib import nullcontext

import torch
from torch import nn


class AttentionPool(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, values, padding_mask=None):
        query = self.query.expand(values.shape[0], -1, -1)
        pooled, _ = self.attention(query, values, values, key_padding_mask=padding_mask, need_weights=False)
        return pooled[:, 0]


class KneeModel(nn.Module):
    def __init__(
        self,
        backbone=None,
        backbone_path: str | None = None,
        timm_name: str | None = None,
        feature_dim: int | None = None,
        num_labels: int = 12,
        attention_heads: int = 4,
        gradient_checkpointing: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        if backbone is None and backbone_path:
            from transformers import Dinov2Model

            backbone = Dinov2Model.from_pretrained(backbone_path, local_files_only=True)
            feature_dim = feature_dim or backbone.config.hidden_size
        elif backbone is None and timm_name:
            import timm

            backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
            feature_dim = feature_dim or backbone.num_features
        if backbone is None or feature_dim is None:
            raise ValueError("Indica backbone + feature_dim, backbone_path o timm_name")

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.plane_embedding = nn.Embedding(3, feature_dim)
        self.fluid_embedding = nn.Embedding(2, feature_dim)
        self.fat_embedding = nn.Embedding(2, feature_dim)
        self.slice_pool = AttentionPool(feature_dim, attention_heads)
        self.series_pool = AttentionPool(feature_dim, attention_heads)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_labels),
        )
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        elif gradient_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

    def freeze_backbone(self, frozen: bool = True):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    def autocast_context(self, device):
        if torch.device(device).type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _encode(self, images):
        output = self.backbone(pixel_values=images) if hasattr(self.backbone, "config") else self.backbone(images)
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim == 4:
            output = output.mean(dim=(-2, -1))
        return output

    def forward(self, images, series_meta, series_mask):
        batch, n_series, n_slices, channels, height, width = images.shape
        features = self._encode(images.reshape(-1, channels, height, width))
        features = features.reshape(batch * n_series, n_slices, self.feature_dim)
        series_features = self.slice_pool(features).reshape(batch, n_series, self.feature_dim)
        series_features = series_features + self.plane_embedding(series_meta[..., 0])
        series_features = series_features + self.fluid_embedding(series_meta[..., 1])
        series_features = series_features + self.fat_embedding(series_meta[..., 2])
        study_features = self.series_pool(series_features, padding_mask=~series_mask.bool())
        return self.head(study_features)
