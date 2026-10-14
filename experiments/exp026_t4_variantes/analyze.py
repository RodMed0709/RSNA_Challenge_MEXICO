#!/usr/bin/env python3
"""exp026_t4_variantes -- prediction agreement + AUC per variant from the Kaggle kernel outputs.

  python analyze.py --out-dir <kernel output dir> --studies exp026_studies.csv --metrics <metrics.json>

Agreement vs V0 (warm, disk run): Pearson of the 150x12 probabilities, mean / max |dp|.
AUC: gold (58, train.csv labels) macro AUC and pseudo AUC vs L2 > 0.5 (150 main and the 92 of fold 0);
paired bootstrap (1000) of dAUC gold vs V0 and vs V1. Absolute AUCs are inflated (Raptor was trained on
all train data); only the differences matter.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score  # noqa: F401 (reference implementation)

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
VARS = ["V0", "V1", "V2", "V3", "V4", "V0b"]
N_PER_MIN = 1300
AUC_PER_MIN = 0.00086


def _auc(y, s):
    """Mann-Whitney AUC with average ranks for ties (== sklearn roc_auc_score)."""
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro_auc(y, p):
    a = []
    for j in range(y.shape[1]):
        m = ~np.isnan(y[:, j]); yy = y[m, j]
        if m.sum() > 1 and 0 < yy.sum() < len(yy):
            a.append(_auc(yy, p[m, j]))
    return float(np.mean(a)) if a else float("nan")


def boot_delta(y, pa, pb, n=1000, seed=0):
    rng = np.random.default_rng(seed); d = []
    N = len(y)
    for _ in range(n):
        i = rng.integers(0, N, N)
        d.append(macro_auc(y[i], pa[i]) - macro_auc(y[i], pb[i]))
    d = np.array(d); d = d[np.isfinite(d)]
    return dict(mean=float(d.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                p_le0=float((d <= 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--studies", required=True)
    ap.add_argument("--metrics", required=True)
    a = ap.parse_args()
    R = json.load(open(os.path.join(a.out_dir, "exp026_results.json")))
    st = pd.read_csv(a.studies); st["StudyInstanceUID"] = st.StudyInstanceUID.astype(str)
    st = st[st.role == "main"].set_index("StudyInstanceUID")
    have = set(pd.read_csv(os.path.join(a.out_dir, "preds_V0_disk.csv")).StudyInstanceUID.astype(str))
    st = st[st.index.isin(have)]                          # smoke runs cover a prefix only
    sids = list(st.index)
    gold = st.fold.values == -1; f0 = st.fold.values == 0
    yg = st[[f"gt_{c}" for c in LAB]].values.astype(float)
    yl = st[[f"l2_{c}" for c in LAB]].values.astype(float)
    yl = np.where(np.isnan(yl), np.nan, (yl > 0.5).astype(float))

    P = {}
    for v in VARS:
        f = os.path.join(a.out_dir, f"preds_{v}_disk.csv")
        if os.path.exists(f):
            d = pd.read_csv(f); d["StudyInstanceUID"] = d.StudyInstanceUID.astype(str)
            P[v] = d.set_index("StudyInstanceUID").loc[sids, LAB].values.astype(float)

    runs = {(r["variant"], r["source"]): r for r in R.get("runs", [])}
    out = dict(exp="exp026_t4_variantes", env=R.get("env"), nproc=R.get("nproc"), n_main=len(sids),
               n_gold=int(gold.sum()), import_s=R.get("import_s"), model_load_s=R.get("model_load_s"),
               serial_cold=R.get("serial"), v0_cold={k: R["V0_cold"][k] for k in
                                                     ("wall", "s_per_study", "steady_s_per_study", "t_feed",
                                                      "autotune_excess_s", "per_gpu", "nvsmi")},
               variants={})
    for v in VARS:
        if v not in P: continue
        rd, rr = runs.get((v, "disk"), {}), runs.get((v, "ram"), {})
        e = dict(s_per_study_disk=rd.get("s_per_study"), steady_s_per_study_disk=rd.get("steady_s_per_study"),
                 s_per_study_ram=rr.get("s_per_study"), steady_s_per_study_ram=rr.get("steady_s_per_study"),
                 t_feed_disk=rd.get("t_feed"), autotune_excess_s=rd.get("autotune_excess_s"),
                 per_gpu_disk=rd.get("per_gpu"), nvsmi_disk=rd.get("nvsmi"))
        e["min_1300_disk_steady"] = (e["steady_s_per_study_disk"] or np.nan) * N_PER_MIN / 60
        p = P[v]
        e["gold_auc"] = macro_auc(yg[gold], p[gold])
        e["pseudo_auc_150"] = macro_auc(yl, p)
        e["pseudo_auc_f0_92"] = macro_auc(yl[f0], p[f0])
        for ref in ("V0", "V1"):
            if ref in P and ref != v:
                q = P[ref]
                e[f"vs_{ref}"] = dict(pearson=float(np.corrcoef(p.ravel(), q.ravel())[0, 1]),
                                      mean_abs_dp=float(np.abs(p - q).mean()), max_abs_dp=float(np.abs(p - q).max()),
                                      d_gold_auc=e["gold_auc"] - macro_auc(yg[gold], q[gold]),
                                      d_pseudo_150=e["pseudo_auc_150"] - macro_auc(yl, q),
                                      boot_gold=boot_delta(yg[gold], p[gold], q[gold]),
                                      boot_pseudo_150=boot_delta(yl, p, q))
        out["variants"][v] = e
    # efficiency trade-off vs V1: minutes saved * 0.00086 must exceed the AUC loss
    if "V1" in out["variants"]:
        m1 = out["variants"]["V1"]["min_1300_disk_steady"]
        for v, e in out["variants"].items():
            e["min_saved_vs_V1"] = m1 - e["min_1300_disk_steady"]
            e["auc_equiv_of_time_vs_V1"] = AUC_PER_MIN * e["min_saved_vs_V1"]
    os.makedirs(os.path.dirname(os.path.abspath(a.metrics)), exist_ok=True)
    json.dump(out, open(a.metrics, "w"), indent=1, default=float)
    hdr = f"{'var':4s} {'s/st disk':>9s} {'steady':>7s} {'ram':>6s} {'min1300':>7s} {'goldAUC':>7s} {'ps150':>6s} {'r_vsV0':>8s} {'|dp|':>7s} {'dGold':>7s} {'dPs150':>7s}"
    print(hdr)
    for v, e in out["variants"].items():
        c = e.get("vs_V0", {})
        print(f"{v:4s} {e['s_per_study_disk'] or 0:9.3f} {e['steady_s_per_study_disk'] or 0:7.3f} "
              f"{e['steady_s_per_study_ram'] or 0:6.3f} {e['min_1300_disk_steady']:7.1f} {e['gold_auc']:7.4f} "
              f"{e['pseudo_auc_150']:6.4f} {c.get('pearson', 1):8.5f} {c.get('mean_abs_dp', 0):7.4f} "
              f"{c.get('d_gold_auc', 0):+7.4f} {c.get('d_pseudo_150', 0):+7.4f}")


if __name__ == "__main__":
    main()
