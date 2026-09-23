from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .dicom import read_series


def _cache_one(args):
    record, raw_root, cache_root, image_size = args
    study = str(record["StudyInstanceUID"])
    series = str(record["SeriesInstanceUID"])
    output = Path(cache_root) / study / f"{series}.npy"
    base = {
        "study": study,
        "series": series,
        "plane": record["Anatomical_Plane"],
        "fluid": int(record["Fluid_Sensitive"]),
        "fat": int(record["Fat_Suppression"]),
        "path": str(output),
    }
    try:
        if output.exists():
            cached = np.load(output, mmap_mode="r")
            return {**base, "n_slices": int(cached.shape[0]), "error": ""}
        volume = read_series(Path(raw_root) / study / series)
        resized = np.stack([
            cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            for image in volume
        ]).astype(np.uint8)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, resized)
        return {**base, "n_slices": len(resized), "error": ""}
    except Exception as exc:
        return {**base, "n_slices": 0, "path": "", "error": f"{type(exc).__name__}: {exc}"}


def build_cache(
    series_csv: str | Path,
    raw_series_root: str | Path,
    cache_root: str | Path,
    image_size: int = 256,
    workers: int = 8,
) -> pd.DataFrame:
    series_df = pd.read_csv(series_csv) if not isinstance(series_csv, pd.DataFrame) else series_csv.copy()
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    jobs = [(record, str(raw_series_root), str(cache_root), image_size) for record in series_df.to_dict("records")]

    if workers == 1:
        rows = [_cache_one(job) for job in tqdm(jobs, desc="Series")]
    else:
        with Pool(processes=workers) as pool:
            rows = list(tqdm(pool.imap_unordered(_cache_one, jobs), total=len(jobs), desc="Series"))

    index = pd.DataFrame(rows)
    index.to_csv(cache_root / "index.csv", index=False)
    errors = index[index["error"] != ""]
    if len(errors):
        errors.to_csv(cache_root / "errors.csv", index=False)
        print(f"{len(errors)} series fallaron. Revisa {cache_root / 'errors.csv'}")
    print(f"Cache listo: {len(index) - len(errors)}/{len(index)} series")
    return index
