import hashlib
import json

import pandas as pd

from .config import SPLITS_ROOT

GOLD_FOLD = -1
EXCLUDED_FOLD = -2


def load_split(name="split_v1_seed42", root=SPLITS_ROOT):
    csv_path = root / f"{name}.csv"
    info = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if digest != info["csv_sha256"]:
        raise RuntimeError(f"{csv_path.name} no coincide con su hash. Los splits no se editan; vuelve a la versión de git.")
    split = pd.read_csv(csv_path)
    split["StudyInstanceUID"] = split["StudyInstanceUID"].astype(str)
    assert split.StudyInstanceUID.is_unique
    assert (split.loc[split.is_gold == 1, "fold"] == GOLD_FOLD).all()
    assert split.loc[split.fold >= 0].groupby("group_id").fold.nunique().max() == 1
    split.attrs.update(name=name, sha256=digest, n_folds=info["n_folds"])
    return split
