"""exp054 phase 1: L5 health check without gold (gold only reported). Usage: python validar_L5.py DATA_DIR L5_CSV OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

data, l5_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def read(p):
    d = pd.read_csv(p, low_memory=False)
    d["StudyInstanceUID"] = d.StudyInstanceUID.astype(str)
    return d.set_index("StudyInstanceUID")[LABELS].apply(pd.to_numeric, errors="coerce").astype(float)


train = pd.read_csv(data / "train.csv", encoding="utf-8")
train["StudyInstanceUID"] = train.StudyInstanceUID.astype(str)
gold = train.dropna(subset=LABELS, how="all").set_index("StudyInstanceUID")[LABELS].astype(float)
assert len(gold) == 58
L5 = read(l5_path)
refs = {"L2": read(data / "L2.csv"), "L3": read(data / "L3.csv"), "L4": read(data / "L4.csv"),
        "raptor": read(data / "raptor.csv")}
allids = sorted(set().union(*[r.index for r in refs.values()]))
stack = np.stack([r.reindex(allids).to_numpy() for r in refs.values()])
n = np.isfinite(stack).sum(0)
refs["consenso4"] = pd.DataFrame(np.where(n > 0, np.nansum(stack, 0) / np.maximum(n, 1), np.nan),
                                 index=allids, columns=LABELS)

complete = L5.notna().all(axis=1)
res = {"n_train": len(train), "coverage_complete": int(complete.reindex(train.StudyInstanceUID).fillna(False).sum())}
res["coverage_pct"] = round(100 * res["coverage_complete"] / len(train), 3)
pool = [u for u in train.StudyInstanceUID if u not in gold.index and complete.get(u, False)]
res["n_pool"] = len(pool)
res["L5_mean"] = round(float(L5.loc[pool].values.mean()), 4)
res["L5_pos_rate"] = {l: round(float((L5.loc[pool, l] > 0.5).mean()), 4) for l in LABELS}


def agree(ref, ids):
    per, kap = {}, {}
    for l in LABELS:
        r = ref.reindex(ids)[l]; p = L5.reindex(ids)[l]
        m = r.notna() & p.notna()
        y = (r[m] > 0.5).astype(int)
        per[l] = round(float(roc_auc_score(y, p[m])), 4) if y.nunique() == 2 else None
        kap[l] = round(float(cohen_kappa_score(y, (p[m] > 0.5).astype(int))), 4)
    v = [x for x in per.values() if x is not None]
    return {"macro_auc": round(float(np.mean(v)), 4), "macro_kappa": round(float(np.mean(list(kap.values()))), 4),
            "auc": per, "kappa": kap}


res["vs"] = {name: agree(ref, pool) for name, ref in refs.items()}
# gold: report only
g = {}
for l in LABELS:
    y = gold[l].dropna(); p = L5.reindex(y.index)[l]; m = p.notna()
    g[l] = round(float(roc_auc_score(y[m], p[m])), 4) if y[m].nunique() == 2 else None
gv = [x for x in g.values() if x is not None]
res["gold_report_only"] = {"macro_auc": round(float(np.mean(gv)), 4), "n_labels": len(gv), "auc": g}
for name, ref in refs.items():
    gg = []
    for l in LABELS:
        y = gold[l].dropna(); p = ref.reindex(y.index)[l]; m = p.notna()
        if y[m].nunique() == 2:
            gg.append(roc_auc_score(y[m], p[m]))
    res["gold_report_only"][f"ref_{name}_macro_auc"] = round(float(np.mean(gg)), 4)

# inter-prompt agreement
stem = l5_path.with_suffix("")
P = {}
for p in ["A_extract", "B_reason", "C_paraphrase"]:
    f = Path(f"{stem}__{p}.csv")
    if f.is_file():
        P[p] = read(f).reindex(pool)
ip = {}
names = list(P)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = P[names[i]], P[names[j]]
        ip[f"{names[i]}-{names[j]}"] = round(float(np.mean([cohen_kappa_score(a[l] > 0.5, b[l] > 0.5) for l in LABELS])), 4)
res["inter_prompt_macro_kappa"] = ip
res["per_prompt_macro_auc_vs_consenso4"] = {}
for p, df in P.items():
    v = []
    for l in LABELS:
        y = (refs["consenso4"].reindex(pool)[l] > 0.5).astype(int)
        if y.nunique() == 2:
            v.append(roc_auc_score(y, df[l]))
    res["per_prompt_macro_auc_vs_consenso4"][p] = round(float(np.mean(v)), 4)

vs = res["vs"]
min_lab = min(x for x in vs["consenso4"]["auc"].values() if x is not None)
checks = {
    "coverage>=99%": res["coverage_pct"] >= 99,
    "macroAUC>=0.90 vs L2,L3,L4,raptor": all(vs[k]["macro_auc"] >= 0.90 for k in ["L2", "L3", "L4", "raptor"]),
    "macroAUC>=0.93 vs consenso4": vs["consenso4"]["macro_auc"] >= 0.93,
    "min label AUC>=0.80 vs consenso4": min_lab >= 0.80,
}
res["health_checks"] = checks
res["healthy_excluding_cost"] = all(checks.values())
out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps({k: res[k] for k in ["coverage_pct", "n_pool", "L5_mean", "health_checks", "healthy_excluding_cost",
                                       "inter_prompt_macro_kappa", "per_prompt_macro_auc_vs_consenso4"]}, indent=1))
for k, v in vs.items():
    print(k, "macroAUC", v["macro_auc"], "macroK", v["macro_kappa"])
print("consenso4 per-label AUC", vs["consenso4"]["auc"])
print("gold (report only)", res["gold_report_only"])
