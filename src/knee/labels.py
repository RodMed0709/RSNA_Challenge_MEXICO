import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import LABELS


def build_report_prompt(report: str) -> str:
    keys = ", ".join(LABELS)
    return f"""Read this knee MRI report, in any language, and return only one JSON object.
Use exactly these keys: {keys}.
Values: 1 if present, 0 if absent, 0.5 if uncertain or not discussed. Do not infer findings.

REPORT:
{report}
"""


def parse_label_json(text: str) -> dict[str, float]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("La respuesta no contiene un objeto JSON")
    payload = json.loads(match.group(0))
    normalized = {str(key).strip().lower().replace("’", "'"): value for key, value in payload.items()}
    result = {}
    for label in LABELS:
        key = label.lower()
        if key not in normalized:
            raise ValueError(f"Falta la etiqueta {label}")
        value = normalized[key]
        if isinstance(value, str):
            value = {"present": 1, "absent": 0, "uncertain": 0.5}.get(value.lower(), value)
        value = float(value)
        if value not in (0.0, 0.5, 1.0):
            raise ValueError(f"Valor inválido para {label}: {value}")
        result[label] = value
    return result


def evaluate_pseudo_labels(gold: pd.DataFrame, pseudo: pd.DataFrame) -> pd.DataFrame:
    merged = gold[["StudyInstanceUID", *LABELS]].merge(
        pseudo[["StudyInstanceUID", *LABELS]], on="StudyInstanceUID", suffixes=("_gold", "_pseudo")
    )
    rows = []
    for label in LABELS:
        y_true = pd.to_numeric(merged[f"{label}_gold"], errors="coerce")
        y_score = pd.to_numeric(merged[f"{label}_pseudo"], errors="coerce")
        valid = y_true.notna() & y_score.notna()
        agreement = float((y_true[valid] == y_score[valid]).mean()) if valid.any() else np.nan
        auc = np.nan
        if valid.any() and y_true[valid].nunique() > 1:
            auc = float(roc_auc_score(y_true[valid], y_score[valid]))
        rows.append({"label": label, "n": int(valid.sum()), "agreement": agreement, "auc": auc})
    return pd.DataFrame(rows)
