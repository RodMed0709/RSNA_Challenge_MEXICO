"""Aggregate metric primitives for exp091_stress_v1."""

from __future__ import annotations

import math
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np


UID_RE = re.compile(r"^\d+(?:\.\d+){3,}$")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = y == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = _average_ranks(score)[positives].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auc_filtered(y: np.ndarray, pred: np.ndarray, min_class_count: int = 30) -> dict[str, Any]:
    """Macro AUC using only labels that satisfy both class-count gates."""

    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if y.shape != pred.shape or y.ndim != 2:
        raise ValueError("y and pred must be two-dimensional arrays with equal shape")
    aucs: list[float] = []
    counts: list[list[int]] = []
    included: list[int] = []
    for label in range(y.shape[1]):
        valid = np.isfinite(y[:, label]) & np.isin(y[:, label], (0.0, 1.0))
        target = y[valid, label]
        n_neg = int((target == 0).sum())
        n_pos = int((target == 1).sum())
        if n_neg < min_class_count or n_pos < min_class_count:
            continue
        aucs.append(_binary_auc(target, pred[valid, label]))
        counts.append([n_neg, n_pos])
        included.append(label)
    return {
        "auc": float(np.mean(aucs)) if aucs else None,
        "n_labels": len(aucs),
        "included_labels": included,
        "class_counts": counts,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx, ry = _average_ranks(np.asarray(x, dtype=float)), _average_ranks(np.asarray(y, dtype=float))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def stability_metrics(reference: np.ndarray, condition: np.ndarray) -> dict[str, Any]:
    """Label-free stability: global MAD and mean per-label Spearman."""

    reference = np.asarray(reference, dtype=float)
    condition = np.asarray(condition, dtype=float)
    if reference.shape != condition.shape or reference.ndim != 2:
        raise ValueError("reference and condition must be equal two-dimensional arrays")
    rhos = [_spearman(reference[:, j], condition[:, j]) for j in range(reference.shape[1])]
    finite = [rho for rho in rhos if np.isfinite(rho)]
    return {
        "mad": float(np.mean(np.abs(condition - reference))),
        "rho": float(np.mean(finite)) if finite else None,
        "n": int(reference.shape[0]),
        "n_labels": len(finite),
    }


def select_hard_cases(y: np.ndarray, pred: np.ndarray, fraction: float = 0.1):
    """Select highest masked per-study BCE; 0.5/NaN targets are excluded."""

    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if y.shape != pred.shape or y.ndim != 2:
        raise ValueError("y and pred must be equal two-dimensional arrays")
    valid = np.isfinite(y) & np.isin(y, (0.0, 1.0))
    clipped = np.clip(pred, 1e-7, 1 - 1e-7)
    term = -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
    term[~valid] = 0.0
    counts = valid.sum(axis=1)
    losses = np.divide(term.sum(axis=1), counts, out=np.full(len(y), np.nan), where=counts > 0)
    eligible = np.flatnonzero(np.isfinite(losses))
    count = max(1, int(math.ceil(len(eligible) * fraction)))
    chosen = eligible[np.argsort(-losses[eligible], kind="mergesort")[:count]]
    return np.sort(chosen), losses


def paired_mean_delta(reference: np.ndarray, candidate: np.ndarray, bootstrap: int = 1000, seed: int = 42):
    """Candidate minus reference mean with a paired study bootstrap."""

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    valid = np.isfinite(reference) & np.isfinite(candidate)
    delta = candidate[valid] - reference[valid]
    if not len(delta):
        raise ValueError("no finite pairs")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(bootstrap, len(delta)))
    boot = delta[indices].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n": int(len(delta)),
    }


def validate_public_metrics(payload: Any) -> None:
    """Reject identifiers, undersized metric cells and non-finite JSON numbers."""

    def visit(value: Any, path: str = "root") -> None:
        if isinstance(value, dict):
            for key in value:
                lowered = str(key).lower()
                harmless_uid_flag = lowered == "contains_uids" and value[key] is False
                if ("uid" in lowered and not harmless_uid_flag) or lowered in {"predictions", "study_ids", "manifests"}:
                    raise ValueError(f"UID or per-study data is forbidden at {path}.{key}")
            if "n" in value and isinstance(value["n"], (int, float)) and value["n"] < 20:
                raise ValueError(f"published metric cells require n >= 20 at {path}")
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, str) and UID_RE.fullmatch(value):
            raise ValueError(f"UID value is forbidden at {path}")
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError(f"non-finite number at {path}")

    visit(payload)


def atomic_write_public_metrics(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    validate_public_metrics(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="exp091 aggregate stress analysis")
    parser.add_argument("--config", default=str(Path(__file__).with_name("panel_config.json")))
    parser.add_argument("--results", default=str(Path(__file__).resolve().parents[2] / "results" / "exp091_stress_v1"))
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.initialize:
        payload = {
            "schema_version": 1,
            "exp": "exp091_stress_v1",
            "phase": "partial",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "planned": {panel: len(config["panels"][panel]) for panel in ("R", "F", "S")},
            "completed": {"R": [], "F": [], "S": []},
            "derived_ensembles": [entry["id"] for entry in config.get("derived_ensembles", [])],
            "privacy": {"minimum_cell_n": 20, "contains_uids": False},
        }
        path = atomic_write_public_metrics(Path(args.results) / "metrics.json", payload)
        print(path)
        return 0
    parser.error("one analysis mode is required")


if __name__ == "__main__":
    raise SystemExit(main())
