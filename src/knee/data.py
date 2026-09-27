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


SELECTORS = ("priority_v1", "slots_v2")
MIRROR_TO = "R"  # mirror=True lleva todo a rodilla derecha: se espejan los estudios "L"; "unknown" no se toca
SLOT1_FALLBACK = 4  # slots_v2: si falta el slot 1 (SAG líquido FS) se usa el slot 4 (SAG PD/T2 sin FS) en su lugar


def attach_meta_v2(index: pd.DataFrame, meta_dir: str | Path) -> pd.DataFrame:
    """Añade slot, side y weighting de meta_v2 a un índice de cache sin ellos (p. ej. v1_256)."""
    meta_dir = Path(meta_dir)
    studies = pd.read_csv(meta_dir / "study_meta_v2.csv")
    series = pd.read_csv(meta_dir / "series_meta_v2.csv", usecols=["series", "weighting"])
    slot_cols = [c for c in studies.columns if c[:4] == "slot" and c[4:5].isdigit() and not c.endswith("_fallback")]
    slots = pd.concat([
        studies[[col]].dropna().rename(columns={col: "series"}).assign(slot=int(col[4])) for col in slot_cols
    ])
    index = index.drop(columns=[c for c in ("slot", "side", "weighting") if c in index])
    index = index.merge(slots, on="series", how="left").merge(series, on="series", how="left")
    side = studies.set_index("study")["study_side"]
    return index.assign(side=index["study"].astype(str).map(side).fillna("unknown"))


def select_slots(frame: pd.DataFrame, max_series: int) -> pd.DataFrame:
    """Una serie por slot; posición = slot - 1. Respaldo: slot 1 vacío -> serie del slot 4 (el 4 queda vacío)."""
    frame = frame[frame["slot"].notna()].sort_values("slot").drop_duplicates("slot")
    position = frame["slot"].astype(int) - 1
    if 1 not in set(frame["slot"].astype(int)):
        position = position.where(frame["slot"].astype(int) != SLOT1_FALLBACK, 0)
    return frame.assign(_position=position)[position < max_series]


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
        max_series: int | None = None,
        n_slices: int = 16,
        image_size: int = 224,
        train: bool = False,
        selector: str = "priority_v1",
        mirror: bool = False,
        crop_mm: float | None = None,
    ):
        """selector="priority_v1" (5 series por prioridad plano/líquido) o "slots_v2" (6 posiciones fijas, una por
        slot de meta_v2; necesita columna `slot`). mirror=True espeja los estudios `side == "L"` a derecha
        (columna `mirror_axis`, o por plano: sagital invierte el orden de cortes, coronal/axial voltea columnas).
        crop_mm recorta un cuadrado centrado de crop_mm / spacing_mm px antes del resize (cache v2, ±10 mm en train)."""
        if selector not in SELECTORS:
            raise ValueError(f"selector debe ser uno de {SELECTORS}")
        if max_series is None:
            max_series = 6 if selector == "slots_v2" else 5
        self.selector = selector
        self.mirror = mirror
        self.crop_mm = crop_mm
        self.studies = [str(study) for study in studies]
        index = pd.read_csv(cache_index) if not isinstance(cache_index, pd.DataFrame) else cache_index.copy()
        if "error" in index:
            index = index[index["error"].fillna("") == ""]
        self.by_study = {str(study): frame.copy() for study, frame in index.groupby("study")}
        for column, needed in (("slot", selector == "slots_v2"), ("side", mirror), ("spacing_mm", crop_mm)):
            if needed and column not in index:
                raise ValueError(f"El índice no tiene la columna {column!r} (usa el cache v2 o attach_meta_v2)")
        for study, frame in self.by_study.items():
            if selector == "slots_v2":
                self.by_study[study] = select_slots(frame, max_series)
            else:
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

    def _load_volume(self, row) -> np.ndarray:
        volume = np.load(row["path"], mmap_mode="r")
        if self.mirror and row["side"] != MIRROR_TO and row["side"] in ("L", "R"):
            axis = row.get("mirror_axis")
            if not isinstance(axis, str):
                axis = "slices" if row["plane"] == "Sagittal" else "cols"
            volume = volume[::-1] if axis == "slices" else volume[:, :, ::-1]
        if self.crop_mm:
            spacing = float(row["spacing_mm"])
            size = int(round(self.crop_mm / spacing))
            top, left = (volume.shape[1] - size) // 2, (volume.shape[2] - size) // 2
            if self.train:
                jitter = int(round(10 / spacing))
                top += np.random.randint(-jitter, jitter + 1)
                left += np.random.randint(-jitter, jitter + 1)
            crop = np.zeros((len(volume), size, size), dtype=volume.dtype)
            r0, c0 = max(top, 0), max(left, 0)
            r1, c1 = min(top + size, volume.shape[1]), min(left + size, volume.shape[2])
            crop[:, r0 - top:r1 - top, c0 - left:c1 - left] = volume[:, r0:r1, c0:c1]
            volume = crop
        return volume

    def __getitem__(self, index):
        study = self.studies[index]
        frame = self.by_study.get(study, pd.DataFrame()).head(self.max_series)
        slots = self.selector == "slots_v2"
        images = torch.zeros(self.max_series, self.n_slices, 3, self.image_size, self.image_size)
        meta = torch.zeros(self.max_series, 3, dtype=torch.long)
        mask = torch.zeros(self.max_series, dtype=torch.bool)
        for series_index, (_, row) in enumerate(frame.iterrows()):
            series_index = int(row["_position"]) if slots else series_index
            volume = self._load_volume(row)
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
