"""exp040 - analisis de robustez: bootstrap pareado, tablas, figuras, subgrupos, calibracion y metrics.json.

Script auxiliar declarado en la SPEC (no entrena). Lee results/exp040_robustez/preds/*.parquet (de infer_robustez.py)
y meta_subgrupos.csv. Escribe en results/exp040_robustez/: robustez_macro.csv, robustez_por_hallazgo.csv,
subgrupos.csv, calibracion.csv, metrics.json y figuras (copias pequenas en docs/figs/exp040_*.png).

Uso: RSNA_USER=rodrigo python experiments/exp040_robustez/analisis_robustez.py
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.knee.config import LABELS, current_user
from src.knee.experiments import git_info, guardar_metricas
from src.knee.metrics import mean_auc
from src.knee.splits import load_split

EXP = "exp040_robustez"
RES = REPO / "results" / EXP
PRED = RES / "preds"
FIGS = REPO / "docs" / "figs"
L2_CSV = "/workspace/data/labels/public/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
V1_CSV = "/workspace/data/labels/report_labels_v1.csv"
B = 1000
SEED = 42

GROUPS = {
    "1 referencia": ["ref", "mirror", "tta"],
    "1b ventanas": ["k6", "k12", "k48", "kall"],
    "2 ruido": ["gauss2", "gauss5", "gauss10", "rician5", "rician10"],
    "3 intensidad": ["gamma70", "gamma140", "contr+20", "contr-20", "bias"],
    "4 cortes": ["cortes50", "cortes25", "shift+2", "shift-2"],
    "5 series": ["drop_s0", "drop_s1", "drop_s2", "drop_s3", "drop_s4", "drop_cor", "drop_sag"],
    "6 resolucion": ["res192", "cropx+16", "cropx-16", "cropy+16", "cropy-16"],
}
DESC = {"ref": "limpia (k24, sin TTA)", "mirror": "espejo anatómico solo", "tta": "TTA media(limpia, espejo)",
        "k6": "6 ventanas", "k12": "12 ventanas", "k48": "48 ventanas", "kall": "todas las ventanas (~94)",
        "gauss2": "ruido gauss σ=2 %", "gauss5": "ruido gauss σ=5 %", "gauss10": "ruido gauss σ=10 %",
        "rician5": "ruido Rician σ=5 %", "rician10": "ruido Rician σ=10 %", "gamma70": "gamma 0.7",
        "gamma140": "gamma 1.4", "contr+20": "contraste +20 %", "contr-20": "contraste −20 %",
        "bias": "bias field exp(±0.2)", "cortes50": "50 % de cortes por slot", "cortes25": "25 % de cortes por slot",
        "shift+2": "desplazamiento +2 cortes", "shift-2": "desplazamiento −2 cortes",
        "drop_s0": "sin Sag fluido (slot 0)", "drop_s1": "sin Sag no fluido (slot 1)",
        "drop_s2": "sin Cor fluido (slot 2)", "drop_s3": "sin Cor (slot 3)", "drop_s4": "sin Axial (slot 4)",
        "drop_cor": "sin plano coronal (2+3)", "drop_sag": "sin plano sagital (0+1)",
        "res192": "384→192→384", "cropx+16": "recorte x+16 px", "cropx-16": "recorte x−16 px",
        "cropy+16": "recorte y+16 px", "cropy-16": "recorte y−16 px"}


def _read(path):
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    df["StudyInstanceUID"] = df.StudyInstanceUID.astype(str)
    return df.set_index("StudyInstanceUID")[LABELS].astype(float)


def boot_weights(n, rng):
    idx = rng.integers(0, n, (B, n))
    W = np.zeros((B, n), np.float64)
    np.add.at(W, (np.repeat(np.arange(B), n), idx.ravel()), 1.0)
    return W


def wauc(y, s, W):
    """AUC ponderada (Mann-Whitney con empates a 0.5) para cada fila de W. y en {0,1}. Devuelve (B,)."""
    o = np.argsort(s, kind="mergesort")
    s, y, W = s[o], y[o], W[:, o]
    starts = np.r_[0, np.nonzero(np.diff(s))[0] + 1]
    Gn = np.add.reduceat(W * (y == 0), starts, axis=1)
    Gp = np.add.reduceat(W * (y == 1), starts, axis=1)
    below = np.cumsum(Gn, axis=1) - Gn
    num = (Gp * (below + 0.5 * Gn)).sum(1)
    den = Gp.sum(1) * Gn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def auc_table(Y, P, W):
    """Y: (n,12) con NaN; P: (n,12); W: (B,n). -> point (12,), boot (B,12)."""
    pt, bt = np.full(12, np.nan), np.full((W.shape[0], 12), np.nan)
    ones = np.ones((1, len(Y)))
    for j in range(12):
        v = np.isfinite(Y[:, j]) & np.isin(Y[:, j], (0.0, 1.0))
        y = Y[v, j]
        if len(np.unique(y)) < 2:
            continue
        pt[j] = wauc(y, P[v, j], ones[:, v])[0]
        bt[:, j] = wauc(y, P[v, j], W[:, v])
    return pt, bt


def ci(x):
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def ece_brier(y, p, bins=10):
    v = np.isfinite(y) & np.isin(y, (0.0, 1.0)); y, p = y[v], p[v]
    e = np.clip((p * bins).astype(int), 0, bins - 1)
    ece = sum(abs(p[e == b].mean() - y[e == b].mean()) * (e == b).sum() for b in range(bins) if (e == b).any()) / len(y)
    return float(ece), float(np.mean((p - y) ** 2)), float(p.mean()), float(y.mean()), int(len(y))


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    split = load_split("split_v1_seed42")
    va = list(split.loc[split.fold == 0, "StudyInstanceUID"].astype(str))
    gold_ids = list(split.loc[split.is_gold == 1, "StudyInstanceUID"].astype(str))
    l2, v1 = _read(L2_CSV), _read(V1_CSV)
    tr = pd.read_csv("/workspace/data/raw/train.csv"); tr["StudyInstanceUID"] = tr.StudyInstanceUID.astype(str)
    gold = tr.dropna(subset=LABELS, how="all").set_index("StudyInstanceUID")[LABELS].astype(float)
    pseudo = (l2 > 0.5).astype(float).where(l2.notna()).reindex(va)
    v1b = v1.where(v1.isin([0.0, 1.0])).reindex(va)
    gold = gold.reindex(gold_ids)
    preds = {p.stem: pd.read_parquet(p) for p in sorted(PRED.glob("*.parquet")) if not p.stem.startswith("smoke")}
    for k in preds:
        preds[k].index = preds[k].index.astype(str)
    if "ref" in preds and "mirror" in preds:
        preds["tta"] = (preds["ref"] + preds["mirror"]) / 2
    conds = [c for g in GROUPS.values() for c in g if c in preds]
    group_of = {c: g for g, cs in GROUPS.items() for c in cs}
    print("condiciones:", len(conds), conds)

    rng = np.random.default_rng(SEED)
    refs = {"pseudo": (pseudo, va, boot_weights(len(va), rng)), "v1": (v1b, va, None), "gold": (gold, gold_ids, boot_weights(len(gold_ids), rng))}
    refs["v1"] = (v1b, va, refs["pseudo"][2])          # mismas réplicas que pseudo (mismos estudios)

    # sanity: el AUC puntual coincide con src.knee.metrics.mean_auc
    chk = mean_auc(pseudo.to_numpy(), preds["ref"].loc[va, LABELS].to_numpy())["mean"]

    macro_rows, lab_rows, store = [], [], {}
    for r, (Y, ids, W) in refs.items():
        Yn = Y.loc[ids, LABELS].to_numpy()
        for c in conds:
            pt, bt = auc_table(Yn, preds[c].loc[ids, LABELS].to_numpy(), W)
            store[(r, c)] = (pt, bt)
        pr, br = store[(r, "ref")]
        for c in conds:
            pt, bt = store[(r, c)]
            m, mb = np.nanmean(pt), np.nanmean(bt, 1)
            d, db = m - np.nanmean(pr), mb - np.nanmean(br, 1)
            lo, hi = ci(mb); dlo, dhi = ci(db)
            macro_rows.append({"group": group_of[c], "cond": c, "desc": DESC[c], "ref": r, "auc": m, "lo": lo, "hi": hi,
                               "delta": d, "dlo": dlo, "dhi": dhi, "n": len(ids)})
            for j, lab in enumerate(LABELS):
                dl, dlh = ci(bt[:, j] - br[:, j])
                alo, ahi = ci(bt[:, j])
                lab_rows.append({"cond": c, "ref": r, "label": lab, "auc": pt[j], "lo": alo, "hi": ahi,
                                 "delta": pt[j] - pr[j], "dlo": dl, "dhi": dlh})
    macro = pd.DataFrame(macro_rows); macro.to_csv(RES / "robustez_macro.csv", index=False)
    lab = pd.DataFrame(lab_rows); lab.to_csv(RES / "robustez_por_hallazgo.csv", index=False)
    ps = macro[macro.ref == "pseudo"].set_index("cond")
    print(f"ref pseudo {ps.loc['ref', 'auc']:.4f} (mean_auc {chk:.4f})")
    print(macro.pivot_table(index="cond", columns="ref", values="delta").reindex(conds).round(4).to_string())

    # ---------------- figura: forest de Δ pseudo ----------------
    fc = [c for c in conds if c != "ref"]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(fc) + 1.2))
    yy = np.arange(len(fc))[::-1]
    d = ps.loc[fc]
    col = ["#c0392b" if x < -0.01 else "#2c7fb8" for x in d.delta]
    ax.errorbar(d.delta, yy, xerr=[d.delta - d.dlo, d.dhi - d.delta], fmt="none", ecolor="#999", lw=1)
    ax.scatter(d.delta, yy, c=col, s=18, zorder=3)
    ax.axvline(0, color="k", lw=0.8); ax.axvline(-0.01, color="#c0392b", lw=0.8, ls="--")
    ax.set_yticks(yy); ax.set_yticklabels([DESC[c] for c in fc], fontsize=7)
    ax.set_xlabel("Δ macro AUC (pseudo L2, val fold 0) vs limpia · IC95 bootstrap"); ax.tick_params(axis="x", labelsize=7)
    ax.set_title(f"exp040 robustez · Raptor f0 (ref {ps.loc['ref', 'auc']:.3f})", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "delta_forest.png", dpi=110); fig.savefig(FIGS / "exp040_delta_forest.png", dpi=110)
    plt.close(fig)

    # ---------------- figura: heatmap por hallazgo ----------------
    hm = lab[(lab.ref == "pseudo") & (lab.cond != "ref")].pivot(index="cond", columns="label", values="delta").reindex(fc)[LABELS]
    fig, ax = plt.subplots(figsize=(8, 0.25 * len(fc) + 1.5))
    lim = max(0.05, float(np.nanpercentile(np.abs(hm.values), 98)))
    im = ax.imshow(hm.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(fc))); ax.set_yticklabels([DESC[c] for c in fc], fontsize=7)
    fig.colorbar(im, ax=ax, label="Δ AUC (pseudo)"); ax.set_title("Δ AUC por hallazgo vs limpia", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "heatmap_hallazgo.png", dpi=110); fig.savefig(FIGS / "exp040_heatmap_hallazgo.png", dpi=110)
    plt.close(fig)

    # ---------------- calibración ----------------
    cal_rows = []
    for r, (Y, ids, _) in (("pseudo", refs["pseudo"]), ("gold", refs["gold"])):
        for c in ("ref", "tta"):
            P = preds[c].loc[ids, LABELS]
            for lab_ in LABELS:
                e, b, mp, prev, n = ece_brier(Y.loc[ids, lab_].to_numpy(), P[lab_].to_numpy())
                cal_rows.append({"ref": r, "cond": c, "label": lab_, "ece": e, "brier": b, "mean_pred": mp, "prevalence": prev, "n": n})
    cal = pd.DataFrame(cal_rows); cal.to_csv(RES / "calibracion.csv", index=False)
    fig, axs = plt.subplots(3, 4, figsize=(11, 8))
    P = preds["ref"].loc[va, LABELS]
    for ax, lab_ in zip(axs.ravel(), LABELS):
        y = pseudo[lab_].to_numpy(); p = P[lab_].to_numpy(); v = np.isfinite(y); y, p = y[v], p[v]
        e = np.clip((p * 10).astype(int), 0, 9)
        xs = [p[e == b_].mean() for b_ in range(10) if (e == b_).sum() >= 5]
        ys = [y[e == b_].mean() for b_ in range(10) if (e == b_).sum() >= 5]
        ax.plot([0, 1], [0, 1], "k--", lw=0.7); ax.plot(xs, ys, "o-", ms=3, color="#2c7fb8")
        ax2 = ax.twinx(); ax2.hist(p, bins=20, range=(0, 1), color="#ccc", alpha=0.6); ax2.set_yticks([])
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        row = cal[(cal.ref == "pseudo") & (cal.cond == "ref") & (cal.label == lab_)].iloc[0]
        ax.set_title(f"{lab_} · ECE {row.ece:.3f} · Brier {row.brier:.3f}", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.tick_params(labelsize=6)
    fig.suptitle("Fiabilidad (val fold 0, contra L2 bin > 0.5) · limpia sin TTA", fontsize=10)
    fig.tight_layout(); fig.savefig(RES / "fiabilidad.png", dpi=100); fig.savefig(FIGS / "exp040_fiabilidad.png", dpi=100)
    plt.close(fig)

    # ---------------- subgrupos ----------------
    sub_rows = []
    mp_ = RES / "meta_subgrupos.csv"
    if mp_.exists():
        meta = pd.read_csv(mp_, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").reindex(va)
        if meta["age"].notna().any():
            a = pd.to_numeric(meta["age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
            meta["age_bin"] = pd.cut(a, [0, 30, 45, 60, 120], labels=["<30", "30-44", "45-59", "60+"]).astype(str)
        def vendor(x):
            x = str(x).lower()
            for k, v in (("siemens", "Siemens"), ("ge", "GE"), ("philips", "Philips"), ("toshiba", "Canon/Toshiba"), ("canon", "Canon/Toshiba")):
                if x.startswith(k):
                    return v
            return "otro" if x != "nan" else "nan"
        meta["manufacturer"] = meta["manufacturer"].map(vendor)
        meta["field_t"] = meta["field_t"].map(lambda x: f"{x:g} T" if pd.notna(x) else "nan")
        rng2 = np.random.default_rng(SEED)
        for var in ["manufacturer", "field_t", "sex", "age_bin", "lang", "study_side"]:
            if var not in meta:
                continue
            for lvl, grp in meta.groupby(meta[var].astype(str)):
                ids = list(grp.index)
                if len(ids) < 30 or lvl in ("nan", "None"):
                    continue
                W = boot_weights(len(ids), rng2)
                rec = {"var": var, "level": lvl, "n": len(ids)}
                for c in ("ref", "mirror", "tta"):
                    pt, bt = auc_table(pseudo.loc[ids, LABELS].to_numpy(), preds[c].loc[ids, LABELS].to_numpy(), W)
                    if c == "ref":
                        pr_, br_ = pt, bt
                        lo, hi = ci(np.nanmean(bt, 1))
                        rec.update({"auc_pseudo": np.nanmean(pt), "lo": lo, "hi": hi, "n_labels": int(np.isfinite(pt).sum())})
                        pv, _ = auc_table(v1b.loc[ids, LABELS].to_numpy(), preds[c].loc[ids, LABELS].to_numpy(), W[:1])
                        rec["auc_v1"] = np.nanmean(pv)
                    else:
                        dlo, dhi = ci(np.nanmean(bt, 1) - np.nanmean(br_, 1))
                        rec.update({f"d_{c}": np.nanmean(pt) - np.nanmean(pr_), f"d_{c}_lo": dlo, f"d_{c}_hi": dhi})
                sub_rows.append(rec)
    sub = pd.DataFrame(sub_rows); sub.to_csv(RES / "subgrupos.csv", index=False)
    if len(sub):
        print(sub.round(4).to_string())
        fig, ax = plt.subplots(figsize=(6, 0.3 * len(sub) + 1))
        yy = np.arange(len(sub))[::-1]
        ax.errorbar(sub.auc_pseudo, yy, xerr=[sub.auc_pseudo - sub.lo, sub.hi - sub.auc_pseudo], fmt="o", ms=3, color="#2c7fb8", ecolor="#999")
        ax.axvline(ps.loc["ref", "auc"], color="k", lw=0.8, ls="--")
        ax.set_yticks(yy); ax.set_yticklabels([f"{r.var}={r.level} (n={r.n})" for r in sub.itertuples()], fontsize=7)
        ax.set_xlabel("macro AUC pseudo L2 · IC95"); ax.set_title("Subgrupos (val fold 0, limpia)", fontsize=9)
        fig.tight_layout(); fig.savefig(RES / "subgrupos.png", dpi=110); fig.savefig(FIGS / "exp040_subgrupos.png", dpi=110)
        plt.close(fig)

    # ---------------- hipótesis y metrics.json ----------------
    D = ps["delta"]
    drops = {c: float(D[c]) for c in ["drop_s0", "drop_s1", "drop_s2", "drop_s3", "drop_s4"] if c in D}
    calp = cal[(cal.ref == "pseudo") & (cal.cond == "ref")]
    hyp = {
        "H1_noise_le5pct_drop_lt_0.01": bool(all(D.get(c, -1) > -0.01 for c in ["gauss2", "gauss5"])),
        "H2_slices50_drop_lt_0.01": bool(D.get("cortes50", -1) > -0.01),
        "H3_most_important_slot": min(drops, key=drops.get) if drops else None,
        "H3_is_sag_fluid": bool(drops and min(drops, key=drops.get) == "drop_s0"),
        "H4_mean_ece_pseudo": float(calp.ece.mean()),
        "H4_ece_lt_0.10": bool(calp.ece.mean() < 0.10),
    }
    tim = pd.read_csv(RES / "timings.csv").set_index("cond") if (RES / "timings.csv").exists() else None
    table = {c: {"group": group_of[c], "desc": DESC[c],
                 **{f"{r}_auc": float(macro[(macro.ref == r) & (macro.cond == c)].auc.iloc[0]) for r in refs},
                 **{f"{r}_delta": float(macro[(macro.ref == r) & (macro.cond == c)].delta.iloc[0]) for r in refs},
                 "pseudo_delta_ci95": [float(ps.loc[c, "dlo"]), float(ps.loc[c, "dhi"])],
                 "pseudo_auc_ci95": [float(ps.loc[c, "lo"]), float(ps.loc[c, "hi"])],
                 "seconds_929": float(tim.loc[c, "seconds"]) if tim is not None and c in tim.index else None}
             for c in conds}
    split_sha = split.attrs.get("sha256")
    metrics = {
        "schema_version": 1, "exp": EXP, "author": current_user(), "parent": "exp029", "split": "split_v1_seed42",
        "split_sha256": split_sha, "git": git_info(REPO), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "analysis", "select_on": "none (solo inferencia, modelo elegido por pseudo L2 en exp029)",
        "model": "exp029 best_f0.pt (epoch 15)", "n_val": len(va), "n_gold": len(gold_ids), "bootstrap": B,
        "reference": {r: {"auc": float(macro[(macro.ref == r) & (macro.cond == "ref")].auc.iloc[0]),
                          "ci95": [float(macro[(macro.ref == r) & (macro.cond == "ref")].lo.iloc[0]),
                                   float(macro[(macro.ref == r) & (macro.cond == "ref")].hi.iloc[0])]} for r in refs},
        "pseudo_auc_f0": float(ps.loc["ref", "auc"]), "gold_auc": table["ref"]["gold_auc"],
        "exp029_pseudo_auc_f0": 0.8681776547817877,
        "conditions": table, "hypotheses": hyp,
        "calibration_pseudo_ref": {"mean_ece": float(calp.ece.mean()), "mean_brier": float(calp.brier.mean()),
                                   "per_label": calp.set_index("label")[["ece", "brier", "mean_pred", "prevalence"]].round(4).to_dict("index")},
        "subgroups": sub.round(4).to_dict("records") if len(sub) else [],
        "gpu_hours_total": float(tim.seconds.sum() / 3600) if tim is not None else None,
    }
    print("hipótesis:", json.dumps(hyp))
    print("Guardado:", guardar_metricas(RES, metrics))


if __name__ == "__main__":
    main()
