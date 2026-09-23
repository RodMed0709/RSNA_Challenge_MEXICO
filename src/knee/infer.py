from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import LABELS, PLANE_TO_ID
from .data import prepare_slices, series_priority
from .dicom import read_series
from .model import KneeModel


def _study_tensors(study, series_frame, raw_root, max_series, n_slices, image_size):
    frame = series_frame.copy()
    frame["plane"] = frame["Anatomical_Plane"]
    frame["fluid"] = frame["Fluid_Sensitive"]
    frame["fat"] = frame["Fat_Suppression"]
    frame = frame.loc[sorted(frame.index, key=lambda i: series_priority(frame.loc[i]))].head(max_series)
    images = torch.zeros(max_series, n_slices, 3, image_size, image_size)
    meta = torch.zeros(max_series, 3, dtype=torch.long)
    mask = torch.zeros(max_series, dtype=torch.bool)
    for index, (_, row) in enumerate(frame.iterrows()):
        volume = read_series(Path(raw_root) / str(study) / str(row["SeriesInstanceUID"]))
        images[index] = prepare_slices(volume, n_slices, image_size)
        meta[index] = torch.tensor([PLANE_TO_ID[row["plane"]], int(row["fluid"]), int(row["fat"])])
        mask[index] = True
    return images.unsqueeze(0), meta.unsqueeze(0), mask.unsqueeze(0)


def predict_submission(
    studies,
    series_csv,
    raw_series_root,
    checkpoint_paths,
    sample_submission,
    backbone_path=None,
    device=None,
):
    studies = [str(study) for study in studies]
    series_df = pd.read_csv(series_csv) if not isinstance(series_csv, pd.DataFrame) else series_csv
    template = pd.read_csv(sample_submission) if not isinstance(sample_submission, pd.DataFrame) else sample_submission.copy()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ensemble = []
    configs = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["cfg"]
        model = KneeModel(
            backbone_path=backbone_path or cfg.get("backbone_path"), timm_name=cfg.get("timm_name"),
            gradient_checkpointing=False, pretrained=False
        )
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        ensemble.append(model)
        configs.append(cfg)
    first = configs[0]
    shape_keys = ("max_series", "n_slices", "image_size")
    if any(any(cfg.get(key) != first.get(key) for key in shape_keys) for cfg in configs[1:]):
        raise ValueError("Todos los checkpoints del ensemble deben usar max_series, n_slices e image_size iguales")
    predictions = []
    with torch.no_grad():
        for study in studies:
            frame = series_df[series_df["StudyInstanceUID"].astype(str) == study]
            tensors = _study_tensors(
                study, frame, raw_series_root, first.get("max_series", 5),
                first.get("n_slices", 16), first.get("image_size", 224)
            )
            images, meta, mask = [tensor.to(device) for tensor in tensors]
            member_predictions = []
            for model in ensemble:
                with model.autocast_context(device):
                    member_predictions.append(model(images, meta, mask).sigmoid().float().cpu().numpy()[0])
            predictions.append(np.mean(member_predictions, axis=0))
    prediction_df = pd.DataFrame(predictions, columns=LABELS)
    prediction_df.insert(0, "StudyInstanceUID", studies)
    submission = template[["StudyInstanceUID"]].merge(prediction_df, on="StudyInstanceUID", how="left")
    return submission[template.columns]
