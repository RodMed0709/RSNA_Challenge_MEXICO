from pathlib import Path

import cv2
import numpy as np
import pydicom


def _slice_position(ds, fallback: int) -> float:
    try:
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        return float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), normal))
    except (AttributeError, TypeError, ValueError):
        return float(getattr(ds, "InstanceNumber", fallback))


def normalize_uint8(volume: np.ndarray) -> np.ndarray:
    low, high = np.percentile(volume, (1, 99))
    if high <= low:
        return np.zeros(volume.shape, dtype=np.uint8)
    volume = np.clip((volume - low) / (high - low), 0, 1)
    return (volume * 255).round().astype(np.uint8)


def read_series(series_dir: str | Path) -> np.ndarray:
    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No hay archivos DICOM en {series_dir}")

    slices = []
    target_shape = None
    for fallback, path in enumerate(files):
        ds = pydicom.dcmread(path)
        pixels = ds.pixel_array.astype(np.float32)
        pixels = pixels * float(getattr(ds, "RescaleSlope", 1.0))
        pixels += float(getattr(ds, "RescaleIntercept", 0.0))
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = pixels.max() + pixels.min() - pixels
        if target_shape is None:
            target_shape = pixels.shape
        elif pixels.shape != target_shape:
            pixels = cv2.resize(pixels, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        slices.append((_slice_position(ds, fallback), pixels))

    slices.sort(key=lambda item: item[0])
    return normalize_uint8(np.stack([pixels for _, pixels in slices]))
