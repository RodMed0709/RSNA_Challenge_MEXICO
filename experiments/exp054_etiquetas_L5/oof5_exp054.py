"""exp054 x 5 folds (fase confirm): pseudo L2 OOF de 5 folds y gold del ensamble de 5 folds.
Uso: cd /workspace/users/rodrigo/wt_exp054_etiquetas_L5 && /workspace/venv312/bin/python ../p7b/oof5_exp054.py
Escribe p7b/oof5_exp054.json y añade "confirm" + oof_pseudo_auc al metrics.json de fold 0 (backup en metrics_f0_backup.json)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from src.knee.metrics import mean_auc, LABELS

ROOT = Path("/workspace/users/rodrigo")
E = "exp054_etiquetas_L5"
DIRS = {0: ROOT / f"wt_{E}/results/{E}"}
DIRS.update({k: ROOT / f"wt_exp054_f{k}/results/{E}" for k in (1, 2, 3, 4)})
L2_CSV = "/workspace/data/labels/public/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
V1_CSV = "/workspace/data/labels/report_labels_v1.csv"


def _read(path):
    df = pd.read_csv(path)
    df["StudyInstanceUID"] = df.StudyInstanceUID.astype(str)
    return df.set_index("StudyInstanceUID")[LABELS].astype(float)


def _pred(p):
    df = pd.read_csv(p); df["StudyInstanceUID"] = df.StudyInstanceUID.astype(str)
    return df.set_index("StudyInstanceUID")[[f"pred_{l}" for l in LABELS]].set_axis(LABELS, axis=1)


l2 = _read(L2_CSV); pseudo_bin = (l2 > 0.5).astype(float).where(l2.notna())
v1 = _read(V1_CSV); v1_bin = v1.where(v1.isin([0.0, 1.0]))
tr = pd.read_csv("/workspace/data/raw/train.csv"); tr["StudyInstanceUID"] = tr.StudyInstanceUID.astype(str)
gold = tr.dropna(subset=LABELS, how="all").set_index("StudyInstanceUID")[LABELS].astype(float)


def sc(ref, pred):
    ids = [u for u in pred.index if u in ref.index]
    return mean_auc(ref.loc[ids, LABELS].to_numpy(), pred.loc[ids, LABELS].to_numpy()), len(ids)


out = {"per_fold": {}, "paths": {}}
oofs, golds = [], []
for k, d in DIRS.items():
    o, g = _pred(d / f"oof_swa_f{k}.csv"), _pred(d / f"gold_swa_f{k}.csv")
    oofs.append(o); golds.append(g)
    m = json.load(open(d / "metrics.json"))["per_fold"][str(k)]
    (p, n), (gg, _) = sc(pseudo_bin, o), sc(gold, g)
    out["per_fold"][k] = {"pseudo_swa": p["mean"], "n_val": n, "gold_swa": gg["mean"], "pseudo_best": m["pseudo_auc"],
                          "best_epoch": m["best_epoch"], "swa_epochs": m["swa"]["epochs"], "gpu_hours": m["gpu_hours"]}
    out["paths"][k] = {"swa": str(d / f"swa_f{k}.pt"), "oof": str(d / f"oof_swa_f{k}.csv"), "gold": str(d / f"gold_swa_f{k}.csv")}
oof = pd.concat(oofs)
assert not oof.index.duplicated().any(), "OOF con estudios repetidos entre folds"
p, n = sc(pseudo_bin, oof); v, _ = sc(v1_bin, oof)
out["oof"] = {"pseudo_L2": p["mean"], "v1": v["mean"], "n": n, "pseudo_per_label": p["per_label"]}
# OOF con rangos por fold (por si la escala difiere entre folds)
oof_r = pd.concat([o.rank(pct=True) for o in oofs]); pr, _ = sc(pseudo_bin, oof_r)
out["oof"]["pseudo_L2_rank_per_fold"] = pr["mean"]
ens = sum(golds) / len(golds)
ens_r = sum(g.rank(pct=True) for g in golds) / len(golds)
ge, _ = sc(gold, ens); ger, _ = sc(gold, ens_r)
out["gold_ens5"] = {"mean_prob": ge["mean"], "mean_rank": ger["mean"], "per_label": ge["per_label"]}
out["gold_mean_single"] = float(np.mean([v["gold_swa"] for v in out["per_fold"].values()]))
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / "p7b" / "oof5_exp054.json", "w"), indent=1)
mp = DIRS[0] / "metrics.json"
m0 = json.load(open(mp))
assert m0["exp"] == E and m0["per_fold"].get("0"), "metrics.json de fold 0 inesperado"
m0["phase"] = "confirm"
m0["oof_pseudo_auc"] = out["oof"]["pseudo_L2"]
m0["confirm"] = {"oof_pseudo_L2": out["oof"]["pseudo_L2"], "oof_v1": out["oof"]["v1"], "oof_n": out["oof"]["n"],
                 "oof_pseudo_L2_rank_per_fold": out["oof"]["pseudo_L2_rank_per_fold"],
                 "gold_ens5_mean_prob": out["gold_ens5"]["mean_prob"], "gold_ens5_mean_rank": out["gold_ens5"]["mean_rank"],
                 "gold_mean_single": out["gold_mean_single"], "per_fold": out["per_fold"], "paths": out["paths"],
                 "gpu_hours_total": float(sum(v["gpu_hours"] for v in out["per_fold"].values()))}
json.dump(m0, open(mp, "w"), indent=2)
print("metrics.json actualizado (phase confirm):", mp)
