from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import LABELS, PLANE_TO_ID


def series_priority(row) -> tuple:
    plane_rank = {"Sagittal": 0, "Coronal": 1, "Axial": 2}.get(row["plane"], 3)
    sequence_rank = 0 if int(row["fluid"]) else 1
    return sequence_rank, plane_rank


def sample_slice_indices(total: int, count: int, train: bool = False) -> np.ndarray:
    if total <= 0:
        return np.zeros(count, dtype=int)
    positions = np.linspace(0, total - 1, count)
    if train and total > count:
        step = (total - 1) / max(count - 1, 1)
        positions += np.random.uniform(-0.35 * step, 0.35 * step, size=count)
    return np.clip(np.round(positions), 0, total - 1).astype(int)


def prepare_slices(volume: np.ndarray, n_slices: int, image_size: int, train: bool = False) -> torch.Tensor:
    indices = sample_slice_indices(len(volume), n_slices, train)
    images = np.stack([
        cv2.resize(volume[index], (image_size, image_size), interpolation=cv2.INTER_AREA)
        for index in indices
    ]).astype(np.float32)
    if train:
        gain = np.random.uniform(0.9, 1.1)
        offset = np.random.uniform(-8, 8)
        angle = np.random.uniform(-5, 5)
        scale = np.random.uniform(0.97, 1.03)
        matrix = cv2.getRotationMatrix2D((image_size / 2, image_size / 2), angle, scale)
        images = np.stack([
            cv2.warpAffine(image, matrix, (image_size, image_size), borderMode=cv2.BORDER_REFLECT_101)
            for image in images
        ])
        images = np.clip(images * gain + offset, 0, 255)
    images = images[:, None] / 255.0
    images = np.repeat(images, 3, axis=1)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    return torch.from_numpy((images - mean) / std).float()


class KneeDataset(Dataset):
    def __init__(
        self,
        studies,
        cache_index: str | Path | pd.DataFrame,
        labels: str | Path | pd.DataFrame | None = None,
        max_series: int = 5,
        n_slices: int = 16,
        image_size: int = 224,
        train: bool = False,
    ):
        self.studies = [str(study) for study in studies]
        index = pd.read_csv(cache_index) if not isinstance(cache_index, pd.DataFrame) else cache_index.copy()
        if "error" in index:
            index = index[index["error"].fillna("") == ""]
        self.by_study = {str(study): frame.copy() for study, frame in index.groupby("study")}
        for study, frame in self.by_study.items():
            self.by_study[study] = frame.loc[sorted(frame.index, key=lambda i: series_priority(frame.loc[i]))]
        self.max_series = max_series
        self.n_slices = n_slices
        self.image_size = image_size
        self.train = train
        if labels is None:
            self.targets = None
        else:
            labels_df = pd.read_csv(labels) if not isinstance(labels, pd.DataFrame) else labels.copy()
            self.targets = labels_df.set_index("StudyInstanceUID")

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        study = self.studies[index]
        frame = self.by_study.get(study, pd.DataFrame()).head(self.max_series)
        images = torch.zeros(self.max_series, self.n_slices, 3, self.image_size, self.image_size)
        meta = torch.zeros(self.max_series, 3, dtype=torch.long)
        mask = torch.zeros(self.max_series, dtype=torch.bool)
        for series_index, (_, row) in enumerate(frame.iterrows()):
            volume = np.load(row["path"], mmap_mode="r")
            images[series_index] = prepare_slices(volume, self.n_slices, self.image_size, self.train)
            meta[series_index] = torch.tensor([
                PLANE_TO_ID.get(row["plane"], 0), int(row["fluid"]), int(row["fat"])
            ])
            mask[series_index] = True
        if not mask.any():
            raise RuntimeError(f"El estudio {study} no tiene series válidas en el cache")

        target = torch.full((len(LABELS),), float("nan"))
        if self.targets is not None and study in self.targets.index:
            values = pd.to_numeric(self.targets.loc[study, LABELS], errors="coerce").to_numpy(dtype=np.float32)
            target = torch.from_numpy(values)
        return {"study": study, "images": images, "series_meta": meta, "series_mask": mask, "targets": target}
