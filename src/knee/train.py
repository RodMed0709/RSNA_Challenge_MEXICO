import json
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import LABELS, RESULTS_ROOT
from .data import KneeDataset
from .metrics import mean_auc
from .model import KneeModel


def _value(cfg, name, default=None):
    return cfg.get(name, default) if isinstance(cfg, dict) else getattr(cfg, name, default)


def masked_bce(logits, targets):
    valid = torch.isfinite(targets)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.nan_to_num(targets), reduction="none")
    return losses[valid].mean()


def _make_folds(labels_df, run_dir, n_folds=5, seed=42):
    path = run_dir / "folds.csv"
    if path.exists():
        return pd.read_csv(path)
    target_matrix = labels_df[LABELS].apply(pd.to_numeric, errors="coerce").fillna(0.5).to_numpy()
    target_matrix = (target_matrix >= 0.5).astype(int)
    folds = np.full(len(labels_df), -1)
    splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, validation_indices) in enumerate(splitter.split(np.zeros(len(labels_df)), target_matrix)):
        folds[validation_indices] = fold
    result = labels_df[["StudyInstanceUID"]].copy()
    result["fold"] = folds
    result.to_csv(path, index=False)
    return result


@contextmanager
def _file_lock(path, timeout=30):
    lock = Path(f"{path}.lock")
    start = time.time()
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"No se pudo obtener el lock {lock}")
            time.sleep(0.2)
    try:
        yield
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def _append_result(cfg, fold, metrics):
    path = Path(_value(cfg, "results_csv", RESULTS_ROOT / "all_runs.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": os.environ.get("RSNA_USER", "unknown"),
        "run_name": _value(cfg, "run_name"),
        "fold": fold,
        "backbone": _value(cfg, "backbone_path") or _value(cfg, "timm_name"),
        "img_size": _value(cfg, "image_size"),
        "n_slices": _value(cfg, "n_slices"),
        "epochs": _value(cfg, "epochs"),
        "lr": _value(cfg, "lr"),
        "val_auc_mean": metrics["mean"],
        "notes": _value(cfg, "notes", ""),
    }
    row.update({f"auc_{label}": value for label, value in metrics["per_label"].items()})
    with _file_lock(path):
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def train_fold(cfg):
    run_dir = Path(_value(cfg, "run_dir"))
    run_dir.mkdir(parents=True, exist_ok=True)
    labels_df = pd.read_csv(_value(cfg, "labels_csv"))
    folds = _make_folds(labels_df, run_dir, _value(cfg, "n_folds", 5), _value(cfg, "seed", 42))
    labels_df = labels_df.merge(folds, on="StudyInstanceUID")
    fold = int(_value(cfg, "fold", 0))
    train_studies = labels_df.loc[labels_df.fold != fold, "StudyInstanceUID"]
    val_studies = labels_df.loc[labels_df.fold == fold, "StudyInstanceUID"]
    dataset_args = dict(
        cache_index=_value(cfg, "cache_index"), labels=labels_df, max_series=_value(cfg, "max_series", 5),
        n_slices=_value(cfg, "n_slices", 16), image_size=_value(cfg, "image_size", 224)
    )
    train_dataset = KneeDataset(train_studies, train=True, **dataset_args)
    val_dataset = KneeDataset(val_studies, train=False, **dataset_args)
    loader_args = dict(batch_size=_value(cfg, "batch_size", 2), num_workers=_value(cfg, "workers", 4), pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    model = KneeModel(
        backbone_path=_value(cfg, "backbone_path"), timm_name=_value(cfg, "timm_name"),
        gradient_checkpointing=_value(cfg, "gradient_checkpointing", False)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=_value(cfg, "lr", 1e-4), weight_decay=_value(cfg, "weight_decay", 0.01))
    accumulation = _value(cfg, "grad_accum", 1)
    total_steps = max(1, math.ceil(len(train_loader) / accumulation) * _value(cfg, "epochs", 5))
    warmup = int(total_steps * _value(cfg, "warmup_ratio", 0.1))

    def schedule(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best_auc = -np.inf
    history = []
    best_payload = None
    for epoch in range(_value(cfg, "epochs", 5)):
        model.freeze_backbone(epoch < _value(cfg, "freeze_epochs", 0))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)
            meta = batch["series_meta"].to(device, non_blocking=True)
            mask = batch["series_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            with model.autocast_context(device):
                loss = masked_bce(model(images, meta, mask), targets) / accumulation
            loss.backward()
            train_loss += loss.item() * accumulation
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), _value(cfg, "grad_clip", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        predictions, targets, studies = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                with model.autocast_context(device):
                    logits = model(
                        batch["images"].to(device), batch["series_meta"].to(device), batch["series_mask"].to(device)
                    )
                predictions.append(logits.sigmoid().float().cpu().numpy())
                targets.append(batch["targets"].numpy())
                studies.extend(batch["study"])
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        metrics = mean_auc(targets, predictions)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / len(train_loader), "val_auc_mean": metrics["mean"]})
        print(f"Epoch {epoch + 1}: loss={history[-1]['train_loss']:.4f} mean_auc={metrics['mean']:.4f}")
        print(" | ".join(f"{label}: {auc:.4f}" for label, auc in metrics["per_label"].items()))
        selection_auc = metrics["mean"] if np.isfinite(metrics["mean"]) else -np.inf
        if best_payload is None or selection_auc > best_auc:
            best_auc = selection_auc
            best_payload = (predictions, targets, studies, metrics)
            torch.save({"model": model.state_dict(), "cfg": dict(cfg) if isinstance(cfg, dict) else vars(cfg)}, run_dir / "best.pt")

    predictions, targets, studies, metrics = best_payload
    oof = pd.DataFrame(predictions, columns=[f"pred_{label}" for label in LABELS])
    oof.insert(0, "StudyInstanceUID", studies)
    for index, label in enumerate(LABELS):
        oof[f"target_{label}"] = targets[:, index]
    oof.to_csv(run_dir / "oof.csv", index=False)
    output = {"best_auc": metrics["mean"] if best_auc == -np.inf else best_auc, "metrics": metrics, "history": history}
    (run_dir / "metrics.json").write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    _append_result(cfg, fold, metrics)
    return output
