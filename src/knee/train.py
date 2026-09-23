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


def _make_folds(labels_df, run_dir, n_folds=5, seed=42, excluded_studies=None):
    path = run_dir / "folds.csv"
    if path.exists():
        result = pd.read_csv(path)
    else:
        target_matrix = labels_df[LABELS].apply(pd.to_numeric, errors="coerce").fillna(0.5).to_numpy()
        target_matrix = (target_matrix >= 0.5).astype(int)
        folds = np.full(len(labels_df), -1)
        splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_, validation_indices) in enumerate(splitter.split(np.zeros(len(labels_df)), target_matrix)):
            folds[validation_indices] = fold
        result = labels_df[["StudyInstanceUID"]].copy()
        result["fold"] = folds
        result.to_csv(path, index=False)
    if excluded_studies:
        result = result.loc[~result["StudyInstanceUID"].astype(str).isin(excluded_studies)].copy()
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


def _append_result(cfg, fold, pseudo_metrics, gold_metrics, selected_metrics, select_on):
    path = Path(_value(cfg, "results_csv", RESULTS_ROOT / "all_runs.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    backbone = _value(cfg, "backbone")
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": os.environ.get("RSNA_USER", "unknown"),
        "run_name": _value(cfg, "run_name"),
        "fold": fold,
        "backbone": _value(cfg, "backbone_path") or _value(cfg, "timm_name") or type(backbone).__name__,
        "img_size": _value(cfg, "image_size"),
        "n_slices": _value(cfg, "n_slices"),
        "epochs": _value(cfg, "epochs"),
        "lr": _value(cfg, "lr"),
        "backbone_lr": _value(cfg, "backbone_lr"),
        "pseudo_auc": pseudo_metrics["mean"],
        "gold_auc": gold_metrics["mean"] if gold_metrics is not None else float("nan"),
        "select_on": select_on,
        "val_auc_mean": selected_metrics["mean"],
        "notes": _value(cfg, "notes", ""),
    }
    label_metrics = gold_metrics if gold_metrics is not None else pseudo_metrics
    row.update({f"auc_{label}": value for label, value in label_metrics["per_label"].items()})
    with _file_lock(path):
        result = pd.DataFrame([row])
        if path.exists():
            result = pd.concat([pd.read_csv(path), result], ignore_index=True)
        result.to_csv(path, index=False)


def _evaluate(model, loader, device):
    predictions, targets, studies = [], [], []
    with torch.no_grad():
        for batch in loader:
            with model.autocast_context(device):
                logits = model(
                    batch["images"].to(device), batch["series_meta"].to(device), batch["series_mask"].to(device)
                )
            predictions.append(logits.sigmoid().float().cpu().numpy())
            targets.append(batch["targets"].numpy())
            studies.extend(batch["study"])
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    return predictions, targets, studies, mean_auc(targets, predictions)


def _prediction_frame(predictions, targets, studies):
    frame = pd.DataFrame(predictions, columns=[f"pred_{label}" for label in LABELS])
    frame.insert(0, "StudyInstanceUID", studies)
    for index, label in enumerate(LABELS):
        frame[f"target_{label}"] = targets[:, index]
    return frame


def train_fold(cfg):
    run_dir = Path(_value(cfg, "run_dir"))
    run_dir.mkdir(parents=True, exist_ok=True)
    labels_df = pd.read_csv(_value(cfg, "labels_csv"))
    labels_df["StudyInstanceUID"] = labels_df["StudyInstanceUID"].astype(str)
    gold_csv = _value(cfg, "gold_csv")
    gold_df = pd.read_csv(gold_csv) if gold_csv else None
    if gold_df is not None:
        gold_df["StudyInstanceUID"] = gold_df["StudyInstanceUID"].astype(str)
        gold_studies = set(gold_df["StudyInstanceUID"])
        labels_df = labels_df.loc[~labels_df["StudyInstanceUID"].isin(gold_studies)].copy()
    else:
        gold_studies = set()
    folds = _make_folds(
        labels_df, run_dir, _value(cfg, "n_folds", 5), _value(cfg, "seed", 42), gold_studies
    )
    folds["StudyInstanceUID"] = folds["StudyInstanceUID"].astype(str)
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
    gold_loader = None
    if gold_df is not None and len(gold_df):
        gold_dataset = KneeDataset(
            gold_df["StudyInstanceUID"], cache_index=_value(cfg, "cache_index"), labels=gold_df,
            max_series=_value(cfg, "max_series", 5), n_slices=_value(cfg, "n_slices", 16),
            image_size=_value(cfg, "image_size", 224), train=False,
        )
        gold_loader = DataLoader(gold_dataset, shuffle=False, **loader_args)

    select_on = _value(cfg, "select_on", "pseudo")
    if select_on not in {"pseudo", "gold"}:
        raise ValueError("select_on debe ser 'pseudo' o 'gold'")
    if select_on == "gold" and gold_loader is None:
        raise ValueError("select_on='gold' requiere gold_csv con al menos un estudio")

    model = KneeModel(
        backbone=_value(cfg, "backbone"), feature_dim=_value(cfg, "feature_dim"),
        backbone_path=_value(cfg, "backbone_path"), timm_name=_value(cfg, "timm_name"),
        gradient_checkpointing=_value(cfg, "gradient_checkpointing", False)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    lr = _value(cfg, "lr", 1e-4)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = AdamW([
        {"params": list(model.backbone.parameters()), "lr": _value(cfg, "backbone_lr", lr / 10)},
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": lr},
    ], weight_decay=_value(cfg, "weight_decay", 0.01))
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
        pseudo_predictions, pseudo_targets, pseudo_studies, pseudo_metrics = _evaluate(model, val_loader, device)
        gold_payload = _evaluate(model, gold_loader, device) if gold_loader is not None else None
        gold_metrics = gold_payload[3] if gold_payload is not None else None
        selected_metrics = gold_metrics if select_on == "gold" else pseudo_metrics
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "pseudo_auc": pseudo_metrics["mean"],
            "gold_auc": gold_metrics["mean"] if gold_metrics is not None else float("nan"),
            "val_auc_mean": selected_metrics["mean"],
        })
        print(
            f"Epoch {epoch + 1}: loss={history[-1]['train_loss']:.4f} "
            f"pseudo_auc={pseudo_metrics['mean']:.4f} gold_auc={history[-1]['gold_auc']:.4f}"
        )
        selection_auc = selected_metrics["mean"] if np.isfinite(selected_metrics["mean"]) else -np.inf
        if best_payload is None or selection_auc > best_auc:
            best_auc = selection_auc
            best_payload = {
                "pseudo": (pseudo_predictions, pseudo_targets, pseudo_studies, pseudo_metrics),
                "gold": gold_payload,
                "selected_metrics": selected_metrics,
            }
            saved_cfg = dict(cfg) if isinstance(cfg, dict) else vars(cfg).copy()
            saved_cfg.pop("backbone", None)
            torch.save({"model": model.state_dict(), "cfg": saved_cfg}, run_dir / "best.pt")

    pseudo_predictions, pseudo_targets, pseudo_studies, pseudo_metrics = best_payload["pseudo"]
    _prediction_frame(pseudo_predictions, pseudo_targets, pseudo_studies).to_csv(run_dir / "oof.csv", index=False)
    gold_metrics = None
    if best_payload["gold"] is not None:
        gold_predictions, gold_targets, gold_studies, gold_metrics = best_payload["gold"]
        _prediction_frame(gold_predictions, gold_targets, gold_studies).to_csv(
            run_dir / "gold_preds.csv", index=False
        )
    selected_metrics = best_payload["selected_metrics"]
    output = {
        "best_auc": selected_metrics["mean"],
        "pseudo_auc": pseudo_metrics["mean"],
        "gold_auc": gold_metrics["mean"] if gold_metrics is not None else float("nan"),
        "select_on": select_on,
        "metrics": selected_metrics,
        "pseudo_metrics": pseudo_metrics,
        "gold_metrics": gold_metrics,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    _append_result(cfg, fold, pseudo_metrics, gold_metrics, selected_metrics, select_on)
    return output
