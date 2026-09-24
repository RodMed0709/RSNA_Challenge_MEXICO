import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.knee.cache import build_cache
from src.knee.config import LABELS
from src.knee.dicom import read_series
from src.knee.labels import parse_label_json
from src.knee.metrics import mean_auc
from src.knee.model import KneeModel


class TinyBackbone(torch.nn.Module):
    def __init__(self, feature_dim=4):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, feature_dim, kernel_size=1)

    def forward(self, images):
        return self.projection(images).mean(dim=(-2, -1))


def test_tiny_dinov2_forward_backward():
    transformers = pytest.importorskip("transformers")
    config = transformers.Dinov2Config(
        image_size=64,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )
    backbone = transformers.Dinov2Model(config)
    model = KneeModel(backbone=backbone, feature_dim=32, num_labels=len(LABELS))

    images = torch.randn(2, 3, 4, 3, 64, 64)
    series_meta = torch.tensor(
        [[[0, 1, 1], [1, 1, 1], [2, 0, 0]]] * 2, dtype=torch.long
    )
    series_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    logits = model(images, series_meta, series_mask)

    assert logits.shape == (2, len(LABELS))
    logits.mean().backward()
    assert model.head[-1].weight.grad is not None


def test_metrics_skip_single_class_labels():
    targets = np.array([[0, 1, 1], [1, 1, 0], [0.5, 1, 1], [0, 1, 1]], dtype=float)
    scores = np.array([[0.1, 0.2, 0.8], [0.9, 0.4, 0.1], [0.99, 0.5, 0.6], [0.2, 0.3, 0.7]])
    result = mean_auc(targets, scores, labels=["good", "single", "other"])

    assert result["per_label"]["good"] == pytest.approx(1.0)
    assert np.isnan(result["per_label"]["single"])
    assert result["mean"] == pytest.approx(1.0)


def test_parse_label_json():
    payload = {label: (1 if i % 3 == 0 else 0.5 if i % 3 == 1 else 0) for i, label in enumerate(LABELS)}
    text = "Respuesta:\n```json\n" + json.dumps(payload) + "\n```"
    parsed = parse_label_json(text)

    assert list(parsed) == LABELS
    assert set(parsed.values()) == {0.0, 0.5, 1.0}


def test_real_dicom_and_cache(tmp_path):
    sample = os.environ.get("RSNA_SAMPLE_DICOM")
    if not sample:
        pytest.skip("RSNA_SAMPLE_DICOM no está definido")

    series_dir = Path(sample)
    volume = read_series(series_dir)
    assert volume.dtype == np.uint8
    assert volume.ndim == 3
    assert volume.shape[0] > 0

    study_uid = "test-study"
    series_uid = "test-series"
    raw_root = tmp_path / "raw"
    copied_series = raw_root / study_uid / series_uid
    copied_series.mkdir(parents=True)
    for index, source in enumerate(sorted(series_dir.glob("*.dcm"))):
        shutil.copy2(source, copied_series / f"{index:03d}.dcm")
    csv_path = tmp_path / "series.csv"
    pd.DataFrame(
        [{
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid,
            "Fluid_Sensitive": 1,
            "Fat_Suppression": 1,
            "Anatomical_Plane": "Sagittal",
        }]
    ).to_csv(csv_path, index=False)

    cache_root = tmp_path / "cache"
    index = build_cache(csv_path, raw_root, cache_root, image_size=64, workers=1)
    cached = np.load(index.iloc[0]["path"])
    assert cached.shape[1:] == (64, 64)
    assert cached.dtype == np.uint8
