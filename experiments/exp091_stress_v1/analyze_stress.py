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


def render_verdict(metrics: dict) -> str:
    coverage = metrics["coverage"]
    gate = metrics["reproduction_gate"]
    hypotheses = metrics.get("hypotheses", {})
    runtime = metrics.get("runtime", {})
    state = "worked" if coverage.get("complete") and gate.get("passed") else "didnt"
    h1 = hypotheses.get("H1", {})
    h2 = hypotheses.get("H2", {})
    h3 = hypotheses.get("H3", {})
    return (
        "## 3. VERDICT (números de `results/exp091_stress_v1/metrics.json`)\n\n"
        f"**Resultado:** {state} · **Fase alcanzada:** análisis de inferencia S/R/F\n\n"
        f"- Cobertura: {coverage.get('completed', 0)}/{coverage.get('planned', 0)} checkpoints; "
        f"completa={str(bool(coverage.get('complete'))).lower()}.\n"
        f"- Gate exp029: pseudo AUC {gate.get('auc', 0):.6f}; Δ {gate.get('delta', 0):+.9f}; "
        f"passed={str(bool(gate.get('passed'))).lower()}.\n"
        f"- H1: rho(clean, R_plaus)={h1.get('rho', 0):.3f}.\n"
        f"- H2: {h2.get('confirmed', 0)}/{h2.get('tested', 0)} comparaciones confirmadas.\n"
        f"- H3: Δ BCE pareado exp054−exp045 en hard cases (ref)={h3.get('delta_bce_ref', 0):+.6f}.\n"
        f"- Inferencia: {runtime.get('gpu_hours', 0):.3f} GPU-h medidas; coste RunPod estimado "
        f"USD {runtime.get('estimated_runpod_cost_usd', 0):.2f}.\n\n"
        "**Qué aprendimos:** ver H1–H3 y tablas agregadas en `metrics.json`; Panel S es descriptivo e in-sample. "
        "No se calculó correlación con LB ni se usó gold para seleccionar.\n\n"
        "**Siguiente paso sugerido:** conservar estas métricas como diagnóstico; no convertirlas en criterio de "
        "selección sin una validación futura fuera de familia/tiempo.\n"
    )


def update_registry_text(text: str, state: str, gate_auc: float, learning: str) -> str:
    replacement = (
        f"| exp091 | stress_v1 | rodrigo | exp040 | stress | {state} | "
        f"{gate_auc:.4f} (gate ref) | — | — | {learning} |"
    )
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith("| exp091 |")]
    if len(matches) != 1:
        raise ValueError(f"expected one exp091 registry row, found {len(matches)}")
    lines[matches[0]] = replacement
    return "\n".join(lines) + "\n"


def _read_frame(path: Path, labels):
    import pandas as pd

    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "StudyInstanceUID" in frame.columns:
        frame["StudyInstanceUID"] = frame.StudyInstanceUID.astype(str)
        frame = frame.set_index("StudyInstanceUID")
    else:
        frame.index = frame.index.astype(str)
    frame = frame.rename(columns={f"pred_{label}": label for label in labels})
    return frame[list(labels)].astype(float)


def _model_output_dir(results: Path, panel: str, model_id: str) -> Path:
    if panel == "F" and model_id == "exp045_f0":
        return results / "private" / "R" / "exp045"
    if panel == "F" and model_id == "exp054_f0":
        return results / "private" / "R" / "exp054"
    return results / "private" / panel / model_id


def _prediction_set(directory: Path, labels):
    from infer_stress import INFERENCE_CONDITIONS

    paths = {condition: directory / f"{condition}.parquet" for condition in INFERENCE_CONDITIONS}
    if not all(path.is_file() for path in paths.values()):
        return None
    raw = {condition: _read_frame(path, labels) for condition, path in paths.items()}
    published = {condition: frame for condition, frame in raw.items() if condition != "mirror"}
    published["tta_mirror"] = (raw["ref"] + raw["mirror"]) / 2.0
    return published


def _bootstrap_weights(n: int, repetitions: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(repetitions, n))
    weights = np.zeros((repetitions, n), dtype=np.float64)
    np.add.at(weights, (np.repeat(np.arange(repetitions), n), indices.ravel()), 1.0)
    return weights


def _weighted_auc(target, score, weights):
    order = np.argsort(score, kind="mergesort")
    target, score, weights = target[order], score[order], weights[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(score)) + 1]
    negatives = np.add.reduceat(weights * (target == 0), starts, axis=1)
    positives = np.add.reduceat(weights * (target == 1), starts, axis=1)
    below = np.cumsum(negatives, axis=1) - negatives
    numerator = (positives * (below + 0.5 * negatives)).sum(axis=1)
    denominator = positives.sum(axis=1) * negatives.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def _auc_table(target, prediction, weights, min_class_count=30):
    point = np.full(target.shape[1], np.nan)
    boot = np.full((weights.shape[0], target.shape[1]), np.nan)
    for label in range(target.shape[1]):
        valid = np.isfinite(target[:, label]) & np.isin(target[:, label], (0.0, 1.0))
        current = target[valid, label]
        if (current == 0).sum() < min_class_count or (current == 1).sum() < min_class_count:
            continue
        point[label] = _weighted_auc(current, prediction[valid, label], np.ones((1, valid.sum())))[0]
        boot[:, label] = _weighted_auc(current, prediction[valid, label], weights[:, valid])
    return point, boot


def _ci(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [None, None]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def _row_bce(target, prediction):
    valid = np.isfinite(target) & np.isin(target, (0.0, 1.0))
    clipped = np.clip(prediction, 1e-7, 1 - 1e-7)
    terms = -(target * np.log(clipped) + (1 - target) * np.log(1 - clipped))
    terms[~valid] = 0.0
    counts = valid.sum(axis=1)
    return np.divide(terms.sum(axis=1), counts, out=np.full(len(target), np.nan), where=counts > 0)


def _stability_block(predictions, ids):
    from infer_stress import PLAUSIBLE_CONDITIONS, PUBLISHED_CONDITIONS, SEVERE_CONDITIONS

    reference = predictions["ref"].loc[ids].to_numpy()
    conditions = {}
    for condition in PUBLISHED_CONDITIONS:
        conditions[condition] = stability_metrics(reference, predictions[condition].loc[ids].to_numpy())
    return {
        "n": len(ids),
        "conditions": conditions,
        "S_plaus": float(np.mean([conditions[name]["mad"] for name in PLAUSIBLE_CONDITIONS])),
        "S_sev": float(np.mean([conditions[name]["mad"] for name in SEVERE_CONDITIONS])),
    }


def _partial_metrics(config, results: Path, labels, split, l2):
    from infer_stress import PLAUSIBLE_CONDITIONS, SEVERE_CONDITIONS

    completed = {panel: [] for panel in ("R", "F", "S")}
    quick = {panel: {} for panel in ("R", "F", "S")}
    gold_ids = split.loc[split.is_gold == 1, "StudyInstanceUID"].astype(str).tolist()
    for panel in ("S", "R", "F"):
        for model in config["panels"][panel]:
            predictions = _prediction_set(_model_output_dir(results, panel, model["id"]), labels)
            if predictions is None:
                continue
            completed[panel].append(model["id"])
            val_ids = split.loc[split.fold == int(model.get("fold", 0)), "StudyInstanceUID"].astype(str).tolist()
            if panel == "S":
                quick[panel][model["id"]] = {
                    "val": _stability_block(predictions, val_ids),
                    "gold": _stability_block(predictions, gold_ids),
                    "in_sample": {"val": bool(model["val_in_sample"]), "gold": bool(model["gold_in_sample"])},
                }
            else:
                target = (l2 > 0.5).astype(float).where(l2.notna()).loc[val_ids, labels].to_numpy()
                auc = macro_auc_filtered(target, predictions["ref"].loc[val_ids, labels].to_numpy(), 30)
                quick[panel][model["id"]] = {
                    "n": len(val_ids),
                    "ref_auc": auc["auc"],
                    "expected_auc": model["pseudo_auc"],
                    "delta": auc["auc"] - model["pseudo_auc"],
                    "gate_pass": abs(auc["auc"] - model["pseudo_auc"]) < 0.002,
                }
    planned = {panel: len(config["panels"][panel]) for panel in ("R", "F", "S")}
    return {
        "schema_version": 1,
        "exp": "exp091_stress_v1",
        "phase": "partial",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "planned": planned,
        "completed": completed,
        "quick_metrics": quick,
        "derived_ensembles": [entry["id"] for entry in config.get("derived_ensembles", [])],
        "privacy": {"minimum_cell_n": 20, "contains_uids": False},
    }


def _full_analysis(config, results: Path, bootstrap: int = 1000):
    import pandas as pd

    repo = Path(__file__).resolve().parents[2]
    import sys
    sys.path.insert(0, str(repo))
    from src.knee.config import LABELS
    from src.knee.splits import load_split
    from infer_stress import PLAUSIBLE_CONDITIONS, PUBLISHED_CONDITIONS, SEVERE_CONDITIONS

    labels = list(LABELS)
    split = load_split("split_v1_seed42")
    split["StudyInstanceUID"] = split.StudyInstanceUID.astype(str)
    l2 = _read_frame(Path(config["l2_csv"]), labels)
    completed = {panel: [] for panel in ("R", "F", "S")}
    missing = {panel: [] for panel in ("R", "F", "S")}
    panel_r, robust_boot = {}, {}
    for model in config["panels"]["R"]:
        predictions = _prediction_set(_model_output_dir(results, "R", model["id"]), labels)
        if predictions is None:
            missing["R"].append(model["id"])
            continue
        completed["R"].append(model["id"])
        ids = split.loc[split.fold == int(model["fold"]), "StudyInstanceUID"].astype(str).tolist()
        target = (l2 > 0.5).astype(float).where(l2.notna()).loc[ids, labels].to_numpy()
        weights = _bootstrap_weights(len(ids), bootstrap)
        reference_point, reference_boot = _auc_table(target, predictions["ref"].loc[ids, labels].to_numpy(), weights)
        reference_macro = float(np.nanmean(reference_point))
        reference_macro_boot = np.nanmean(reference_boot, axis=1)
        condition_rows, delta_boot = {}, {}
        for condition in PUBLISHED_CONDITIONS:
            point, boot_values = _auc_table(target, predictions[condition].loc[ids, labels].to_numpy(), weights)
            macro = float(np.nanmean(point))
            macro_boot = np.nanmean(boot_values, axis=1)
            deltas = macro_boot - reference_macro_boot
            delta_boot[condition] = deltas
            condition_rows[condition] = {
                "n": len(ids),
                "n_labels": int(np.isfinite(point).sum()),
                "auc": macro,
                "auc_ci95": _ci(macro_boot),
                "delta": macro - reference_macro,
                "delta_ci95": _ci(deltas),
            }
        plaus_boot = np.mean(np.stack([delta_boot[name] for name in PLAUSIBLE_CONDITIONS]), axis=0)
        severe_boot = np.mean(np.stack([delta_boot[name] for name in SEVERE_CONDITIONS]), axis=0)
        panel_r[model["id"]] = {
            "n": len(ids),
            "expected_auc": model["pseudo_auc"],
            "ref_auc": reference_macro,
            "reproduction_delta": reference_macro - model["pseudo_auc"],
            "gate_pass": abs(reference_macro - model["pseudo_auc"]) < 0.002,
            "conditions": condition_rows,
            "R_plaus": float(np.mean([condition_rows[name]["delta"] for name in PLAUSIBLE_CONDITIONS])),
            "R_plaus_ci95": _ci(plaus_boot),
            "R_sev": float(np.mean([condition_rows[name]["delta"] for name in SEVERE_CONDITIONS])),
            "R_sev_ci95": _ci(severe_boot),
            "worst_plaus": float(min(condition_rows[name]["delta"] for name in PLAUSIBLE_CONDITIONS)),
        }
        robust_boot[model["id"]] = {"plaus": plaus_boot, "sev": severe_boot}

    panel_s = {}
    s_predictions = {}
    gold_ids = split.loc[split.is_gold == 1, "StudyInstanceUID"].astype(str).tolist()
    for model in config["panels"]["S"]:
        predictions = _prediction_set(_model_output_dir(results, "S", model["id"]), labels)
        if predictions is None:
            missing["S"].append(model["id"])
            continue
        completed["S"].append(model["id"])
        s_predictions[model["id"]] = predictions
        val_ids = split.loc[split.fold == int(model["fold"]), "StudyInstanceUID"].astype(str).tolist()
        panel_s[model["id"]] = {
            "val": _stability_block(predictions, val_ids),
            "gold": _stability_block(predictions, gold_ids),
            "in_sample": {"val": bool(model["val_in_sample"]), "gold": bool(model["gold_in_sample"])},
        }
    for ensemble in config.get("derived_ensembles", []):
        if not all(member in s_predictions for member in ensemble["members"]):
            continue
        averaged = {}
        for condition in PUBLISHED_CONDITIONS:
            frames = [s_predictions[member][condition] for member in ensemble["members"]]
            averaged[condition] = sum(frames[1:], frames[0].copy()) / len(frames)
        panel_s[ensemble["id"]] = {
            "val": _stability_block(averaged, split.loc[split.fold == 0, "StudyInstanceUID"].astype(str).tolist()),
            "gold": _stability_block(averaged, gold_ids),
            "in_sample": {"val": True, "gold": True},
            "members": list(ensemble["members"]),
        }

    family_predictions = {"exp045": {}, "exp054": {}}
    for model in config["panels"]["F"]:
        predictions = _prediction_set(_model_output_dir(results, "F", model["id"]), labels)
        if predictions is None:
            missing["F"].append(model["id"])
            continue
        completed["F"].append(model["id"])
        family = "exp045" if model["id"].startswith("exp045") else "exp054"
        family_predictions[family][int(model["fold"])] = predictions

    hard_metrics = {}
    if all(len(family_predictions[family]) == 5 for family in ("exp045", "exp054")):
        all_ids = split.loc[split.fold.isin(range(5)), "StudyInstanceUID"].astype(str).tolist()
        hard_target_frame = l2.loc[all_ids, labels].copy()
        hard_target_frame = hard_target_frame.where(~np.isclose(hard_target_frame, 0.5))
        target = hard_target_frame.to_numpy()
        combined = {family: {} for family in family_predictions}
        for family in combined:
            for condition in PUBLISHED_CONDITIONS:
                frames = [family_predictions[family][fold][condition] for fold in range(5)]
                combined[family][condition] = pd.concat(frames).loc[all_ids, labels]
        selected, discovery_losses = select_hard_cases(target, combined["exp045"]["ref"].to_numpy(), fraction=0.1)
        for condition in PUBLISHED_CONDITIONS:
            loss_045 = _row_bce(target, combined["exp045"][condition].to_numpy())[selected]
            loss_054 = _row_bce(target, combined["exp054"][condition].to_numpy())[selected]
            hard_metrics[condition] = paired_mean_delta(loss_045, loss_054, bootstrap=bootstrap, seed=42)
        hard_metrics = {
            "n": int(len(selected)),
            "fraction": 0.1,
            "selection": "top masked BCE from exp045 OOF; L2 0.5/NaN excluded",
            "delta_bce_exp054_minus_exp045": hard_metrics,
        }

    h1 = {"rho": None, "ci95": [None, None], "n_models": len(panel_r), "criterion_abs_lt_0_5": None, "inconclusive": True}
    if len(panel_r) >= 3:
        ids = list(panel_r)
        clean = np.asarray([panel_r[model_id]["ref_auc"] for model_id in ids])
        robust = np.asarray([panel_r[model_id]["R_plaus"] for model_id in ids])
        rho = _spearman(clean, robust)
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(bootstrap):
            chosen = rng.integers(0, len(ids), len(ids))
            value = _spearman(clean[chosen], robust[chosen])
            if np.isfinite(value):
                boots.append(value)
        interval = _ci(boots)
        h1 = {
            "rho": rho,
            "ci95": interval,
            "n_models": len(ids),
            "criterion_abs_lt_0_5": abs(rho) < 0.5,
            "inconclusive": bool(interval[0] is None or (interval[0] <= 0 <= interval[1] and interval[0] <= -0.5 and interval[1] >= 0.5)),
        }

    h2_rows = {}
    for model_id, row in panel_r.items():
        model = next(item for item in config["panels"]["R"] if item["id"] == model_id)
        parent = model.get("parent")
        if not parent or parent not in panel_r:
            continue
        improvement = row["R_sev"] - panel_r[parent]["R_sev"]
        difference_boot = robust_boot[model_id]["sev"] - robust_boot[parent]["sev"]
        interval = _ci(difference_boot)
        clean_delta = row["ref_auc"] - panel_r[parent]["ref_auc"]
        confirmed = improvement > 0.003 and interval[0] is not None and interval[0] > 0 and clean_delta >= -0.005
        h2_rows[model_id] = {
            "parent": parent,
            "R_sev_improvement": improvement,
            "ci95": interval,
            "clean_delta": clean_delta,
            "confirmed": bool(confirmed),
        }
    h2 = {"tested": len(h2_rows), "confirmed": sum(row["confirmed"] for row in h2_rows.values()), "comparisons": h2_rows}
    h3_delta = None
    if hard_metrics:
        h3_delta = hard_metrics["delta_bce_exp054_minus_exp045"]["ref"]["delta"]
    hypotheses = {"H1": h1, "H2": h2, "H3": {"delta_bce_ref": h3_delta, "provisional": True}}

    inference_seconds = 0.0
    timing_files = list((results / "private").glob("[RSF]/*/timings.json"))
    for timing_file in timing_files:
        timing = json.loads(timing_file.read_text(encoding="utf-8"))
        inference_seconds += sum(float(row["seconds"]) for row in timing.values())
    orq = Path("/workspace/users/rodrigo/orq/stress")
    started_path = orq / "pod_started_epoch"
    started = float(started_path.read_text().strip()) if started_path.exists() else datetime.now(timezone.utc).timestamp()
    wall_hours = max(0.0, (datetime.now(timezone.utc).timestamp() - started) / 3600.0)
    hourly_path = orq / "gpu_hourly_usd"
    hourly_usd = float(hourly_path.read_text().strip()) if hourly_path.exists() else 0.72
    runtime = {
        "inference_seconds": inference_seconds,
        "gpu_hours": inference_seconds / 3600.0,
        "pod_wall_hours": wall_hours,
        "hourly_usd": hourly_usd,
        "estimated_runpod_cost_usd": wall_hours * hourly_usd,
    }

    planned = sum(len(config["panels"][panel]) for panel in ("R", "F", "S"))
    done = sum(len(completed[panel]) for panel in completed)
    coverage = {"planned": planned, "completed": done, "complete": done == planned, "by_panel": completed, "missing": missing}
    exp029 = panel_r.get("exp029")
    reproduction = {
        "auc": exp029["ref_auc"] if exp029 else None,
        "expected": next(item["pseudo_auc"] for item in config["panels"]["R"] if item["id"] == "exp029"),
        "delta": exp029["reproduction_delta"] if exp029 else None,
        "passed": bool(exp029 and exp029["gate_pass"]),
    }
    metrics = {
        "schema_version": 1,
        "exp": "exp091_stress_v1",
        "parent": "exp040",
        "phase": "analysis" if coverage["complete"] else "partial",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": "split_v1_seed42",
        "bootstrap": bootstrap,
        "coverage": coverage,
        "reproduction_gate": reproduction,
        "panel_R": panel_r,
        "panel_S": panel_s,
        "panel_F_hard": hard_metrics,
        "hypotheses": hypotheses,
        "lb_table": config["lb_table"],
        "lb_correlation_computed": False,
        "runtime": runtime,
        "privacy": {"minimum_cell_n": 20, "contains_uids": False},
    }
    _write_figures(metrics, Path(__file__).with_name("figures"))
    return metrics


def _write_figures(metrics: dict, directory: Path):
    if not metrics.get("panel_R"):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    names = list(metrics["panel_R"])
    plaus = [metrics["panel_R"][name]["R_plaus"] for name in names]
    severe = [metrics["panel_R"][name]["R_sev"] for name in names]
    figure, axis = plt.subplots(figsize=(9, max(4, 0.3 * len(names))))
    y = np.arange(len(names))
    axis.barh(y - 0.18, plaus, height=0.35, label="R_plaus")
    axis.barh(y + 0.18, severe, height=0.35, label="R_sev")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, names)
    axis.set_xlabel("Mean Δ macro AUC vs ref")
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "panel_r_robustness.png", dpi=120)
    plt.close(figure)
    if metrics.get("panel_S"):
        names = list(metrics["panel_S"])
        plaus = [metrics["panel_S"][name]["val"]["S_plaus"] for name in names]
        severe = [metrics["panel_S"][name]["val"]["S_sev"] for name in names]
        figure, axis = plt.subplots(figsize=(9, max(4, 0.3 * len(names))))
        y = np.arange(len(names))
        axis.barh(y - 0.18, plaus, height=0.35, label="S_plaus MAD")
        axis.barh(y + 0.18, severe, height=0.35, label="S_sev MAD")
        axis.set_yticks(y, names)
        axis.set_xlabel("Mean absolute prediction change")
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / "panel_s_stability.png", dpi=120)
        plt.close(figure)


def _finalize_docs(metrics: dict):
    repo = Path(__file__).resolve().parents[2]
    spec_path = Path(__file__).with_name("SPEC.md")
    spec = spec_path.read_text(encoding="utf-8")
    state = "worked" if metrics["coverage"]["complete"] and metrics["reproduction_gate"]["passed"] else "didnt"
    spec = re.sub(r"\| Estado \| [^|]+ \|", f"| Estado | {state} |", spec, count=1)
    spec = re.sub(r"## 3\. VERDICT[\s\S]*\Z", render_verdict(metrics), spec)
    spec_path.write_text(spec.rstrip() + "\n", encoding="utf-8")
    h1 = metrics["hypotheses"]["H1"].get("rho")
    learning = f"H1 rho={h1:.3f}; coverage {metrics['coverage']['completed']}/{metrics['coverage']['planned']}"
    registry_path = repo / "docs" / "EXPERIMENTOS.md"
    registry_path.write_text(
        update_registry_text(
            registry_path.read_text(encoding="utf-8"),
            state,
            metrics["reproduction_gate"]["auc"],
            learning,
        ),
        encoding="utf-8",
    )
    log_path = Path(__file__).with_name("BITACORA.md")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} — verdict\n\n"
            f"- Estado: {state}; cobertura {metrics['coverage']['completed']}/{metrics['coverage']['planned']}.\n"
            f"- Gate exp029: {metrics['reproduction_gate']['auc']:.9f} "
            f"(Δ {metrics['reproduction_gate']['delta']:+.9f}).\n"
            f"- Coste RunPod estimado: USD {metrics['runtime']['estimated_runpod_cost_usd']:.2f}.\n"
        )


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="exp091 aggregate stress analysis")
    parser.add_argument("--config", default=str(Path(__file__).with_name("panel_config.json")))
    parser.add_argument("--results", default=str(Path(__file__).resolve().parents[2] / "results" / "exp091_stress_v1"))
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--finalize-docs", action="store_true")
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
    if args.partial:
        repo = Path(__file__).resolve().parents[2]
        import sys
        sys.path.insert(0, str(repo))
        from src.knee.config import LABELS
        from src.knee.splits import load_split
        split = load_split("split_v1_seed42")
        split["StudyInstanceUID"] = split.StudyInstanceUID.astype(str)
        l2 = _read_frame(Path(config["l2_csv"]), LABELS)
        payload = _partial_metrics(config, Path(args.results), LABELS, split, l2)
        path = atomic_write_public_metrics(Path(args.results) / "metrics.json", payload)
        print(path)
        return 0
    if args.final:
        payload = _full_analysis(config, Path(args.results), bootstrap=1000)
        path = atomic_write_public_metrics(Path(args.results) / "metrics.json", payload)
        print(path)
        return 0
    if args.finalize_docs:
        metrics = json.loads((Path(args.results) / "metrics.json").read_text(encoding="utf-8"))
        _finalize_docs(metrics)
        print(Path(__file__).with_name("SPEC.md"))
        return 0
    parser.error("one analysis mode is required")


if __name__ == "__main__":
    raise SystemExit(main())
