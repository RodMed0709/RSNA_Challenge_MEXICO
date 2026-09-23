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
from src.knee.train import train_fold


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


def test_train_fold_tracks_pseudo_and_gold_metrics(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    pseudo_studies = [f"pseudo-{index:02d}" for index in range(12)]
    gold_studies = [f"gold-{index:02d}" for index in range(4)]
    index_rows = []
    for index, study in enumerate([*pseudo_studies, *gold_studies]):
        path = cache_dir / f"{study}.npy"
        np.save(path, np.full((2, 8, 8), index, dtype=np.uint8))
        index_rows.append({
            "study": study,
            "path": str(path),
            "plane": "Sagittal",
            "fluid": 1,
            "fat": 1,
        })
    cache_index = cache_dir / "index.csv"
    pd.DataFrame(index_rows).to_csv(cache_index, index=False)

    def label_row(study, index):
        return {
            "StudyInstanceUID": study,
            **{label: (index + label_index) % 2 for label_index, label in enumerate(LABELS)},
        }

    pseudo_csv = tmp_path / "pseudo.csv"
    gold_csv = tmp_path / "gold.csv"
    pd.DataFrame([
        label_row(study, index) for index, study in enumerate([*pseudo_studies, *gold_studies])
    ]).to_csv(pseudo_csv, index=False)
    pd.DataFrame([label_row(study, index) for index, study in enumerate(gold_studies)]).to_csv(
        gold_csv, index=False
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame({
        "StudyInstanceUID": [*pseudo_studies, *gold_studies],
        "fold": [
            *[(index // 2) % 2 for index in range(len(pseudo_studies))],
            *[0 for _ in gold_studies],
        ],
    }).to_csv(run_dir / "folds.csv", index=False)
    results_csv = tmp_path / "all_runs.csv"
    output = train_fold({
        "run_name": "smoke",
        "run_dir": str(run_dir),
        "fold": 0,
        "n_folds": 2,
        "labels_csv": str(pseudo_csv),
        "gold_csv": str(gold_csv),
        "cache_index": str(cache_index),
        "backbone": TinyBackbone(),
        "feature_dim": 4,
        "image_size": 8,
        "max_series": 1,
        "n_slices": 1,
        "batch_size": 2,
        "epochs": 1,
        "workers": 0,
        "select_on": "gold",
        "results_csv": str(results_csv),
    })

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "oof.csv").exists()
    assert (run_dir / "gold_preds.csv").exists()
    assert set(pd.read_csv(run_dir / "oof.csv")["StudyInstanceUID"]).isdisjoint(gold_studies)
    assert set(pd.read_csv(run_dir / "gold_preds.csv")["StudyInstanceUID"]) == set(gold_studies)
    assert np.isfinite(output["history"][0]["pseudo_auc"])
    assert np.isfinite(output["history"][0]["gold_auc"])
    results = pd.read_csv(results_csv)
    assert results.loc[0, "gold_auc"] == pytest.approx(output["gold_auc"])
    assert results.loc[0, "select_on"] == "gold"


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
