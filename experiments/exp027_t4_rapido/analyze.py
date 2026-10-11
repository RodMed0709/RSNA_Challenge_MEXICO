#!/usr/bin/env python3
"""exp027_t4_rapido -- timing table + prediction agreement + AUC per variant from the Kaggle kernel outputs.

  python analyze.py --out-dir <kernel output dir> --studies exp027_studies.csv --metrics <metrics.json>

Per variant (warm disk run of the notebook pipeline on the 150 main studies):
  steady s/study, minutes @1300 = steady*1300/60 + fixed (imports + model load + autotune excess + compile)/60,
  Pearson / mean / max |dp| vs R, macro AUC gold (58) and pseudo vs L2 > 0.5 (150 and 92 of fold 0), paired
  bootstrap (1000) of dAUC vs R, and the relative efficiency score dAUC - 0.00086 * dmin (vs R).
Absolute AUCs are inflated (both models were trained on all train data); only the differences matter.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
VARS = ["R", "A", "B", "B94", "C", "E1", "Rb", "D", "E2", "E2d", "F", "E1b"]
CFG = {"R": "384", "A": "384", "C": "384", "D": "384", "Rb": "384", "B": "224", "B94": "224", "E1": "224",
       "E2": "224", "E2d": "224", "F": "224", "E1b": "224"}
COMPILED = {"D", "E2", "E2d", "F"}
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
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    R = json.load(open(os.path.join(a.out_dir, "exp027_results.json")))
    st = pd.read_csv(a.studies); st["StudyInstanceUID"] = st.StudyInstanceUID.astype(str)
    st = st[st.role == "main"].set_index("StudyInstanceUID")
    have = set(pd.read_csv(os.path.join(a.out_dir, "preds_R_disk.csv")).StudyInstanceUID.astype(str))
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

    runs = {(r["tag"], r["source"]): r for r in R.get("runs", [])}
    imp = R.get("import_s", 0.0); load = R.get("model_load_s", {})
    ngpu = R.get("env", {}).get("ngpu", 2)
    out = dict(exp="exp027_t4_rapido", env=R.get("env"), nproc=R.get("nproc"), n_main=len(sids),
               n_gold=int(gold.sum()), import_s=imp, model_load_s=load, errors=R.get("errors"),
               rt_info=R.get("rt_info"), build_s=R.get("build_s"), warm_first_s=R.get("warm_first_s"),
               cold_variant=R.get("cold_variant"), variants={}, extra_runs={})
    for tag in ("R_cold", f"{R.get('cold_variant')}_cold", "prep_warm"):
        r = runs.get((tag, "disk"))
        if r:
            out["extra_runs"][tag] = {k: r[k] for k in ("n", "wall", "s_per_study", "steady_s_per_study", "t_feed",
                                                        "autotune_excess_s", "per_gpu", "nvsmi", "fails")}
    for v in VARS:
        if v not in P: continue
        rd, rr = runs.get((v, "disk"), {}), runs.get((v, "ram"), {})
        # model load measured for both GPUs (sequential, as the notebook); compile = main-thread first calls
        compile_s = float(sum(R.get("warm_first_s", {}).get(v if v != "Rb" else "R", {}).values())) if v in COMPILED else 0.0
        fixed = imp + load.get(CFG[v], 0.0) + (rd.get("autotune_excess_s") or 0.0) + compile_s
        e = dict(s_per_study_disk=rd.get("s_per_study"), steady_s_per_study_disk=rd.get("steady_s_per_study"),
                 s_per_study_ram=rr.get("s_per_study"), steady_s_per_study_ram=rr.get("steady_s_per_study"),
                 t_feed_disk=rd.get("t_feed"), autotune_excess_s=rd.get("autotune_excess_s"), compile_s=compile_s,
                 fixed_s=fixed, fails=rd.get("fails"), per_gpu_disk=rd.get("per_gpu"), nvsmi_disk=rd.get("nvsmi"),
                 gpu_s_per_study=np.mean([g.get("steady_gpu_s_per_study", np.nan) for g in (rd.get("per_gpu") or {}).values()]))
        e["min_1300"] = (e["steady_s_per_study_disk"] or np.nan) * N_PER_MIN / 60 + fixed / 60
        e["min_1300_gpu_only"] = (e["steady_s_per_study_ram"] or np.nan) * N_PER_MIN / 60 + fixed / 60
        p = P[v]
        e["gold_auc"] = macro_auc(yg[gold], p[gold])
        e["pseudo_auc_150"] = macro_auc(yl, p)
        e["pseudo_auc_f0_92"] = macro_auc(yl[f0], p[f0])
        if v != "R" and "R" in P:
            q = P["R"]
            e["vs_R"] = dict(pearson=float(np.corrcoef(p.ravel(), q.ravel())[0, 1]),
                             mean_abs_dp=float(np.abs(p - q).mean()), max_abs_dp=float(np.abs(p - q).max()),
                             d_gold_auc=e["gold_auc"] - macro_auc(yg[gold], q[gold]),
                             d_pseudo_150=e["pseudo_auc_150"] - macro_auc(yl, q),
                             d_pseudo_f0_92=e["pseudo_auc_f0_92"] - macro_auc(yl[f0], q[f0]),
                             boot_gold=boot_delta(yg[gold], p[gold], q[gold], a.boot),
                             boot_pseudo_150=boot_delta(yl, p, q, a.boot))
        out["variants"][v] = e
    if "R" in out["variants"]:
        mR = out["variants"]["R"]["min_1300"]
        for v, e in out["variants"].items():
            c = e.get("vs_R", dict(d_gold_auc=0.0, d_pseudo_150=0.0))
            e["d_min_vs_R"] = e["min_1300"] - mR
            e["rel_min_vs_R"] = e["min_1300"] / mR
            e["eff_score_gold"] = c["d_gold_auc"] - AUC_PER_MIN * e["d_min_vs_R"]
            e["eff_score_pseudo"] = c["d_pseudo_150"] - AUC_PER_MIN * e["d_min_vs_R"]
    os.makedirs(os.path.dirname(os.path.abspath(a.metrics)), exist_ok=True)
    json.dump(out, open(a.metrics, "w"), indent=1, default=float)
    print(f"{'var':4s} {'steady':>7s} {'ram':>6s} {'gpu/st':>6s} {'fixed':>6s} {'min1300':>7s} {'rel':>5s} {'gold':>7s} "
          f"{'ps150':>7s} {'r_vsR':>8s} {'|dp|':>7s} {'dGold [IC95]':>26s} {'dPs150':>7s} {'effG':>7s} {'effP':>7s}")
    for v, e in out["variants"].items():
        c = e.get("vs_R", {}); bg = c.get("boot_gold", {})
        ic = f"{c.get('d_gold_auc', 0):+.4f} [{bg.get('lo', 0):+.4f},{bg.get('hi', 0):+.4f}]"
        print(f"{v:4s} {e['steady_s_per_study_disk'] or 0:7.3f} {e['steady_s_per_study_ram'] or 0:6.3f} "
              f"{e['gpu_s_per_study']:6.3f} {e['fixed_s']:6.1f} {e['min_1300']:7.2f} {e['rel_min_vs_R']:5.2f} "
              f"{e['gold_auc']:7.4f} {e['pseudo_auc_150']:7.4f} {c.get('pearson', 1):8.5f} {c.get('mean_abs_dp', 0):7.4f} "
              f"{ic:>26s} {c.get('d_pseudo_150', 0):+7.4f} {e['eff_score_gold']:+7.4f} {e['eff_score_pseudo']:+7.4f}")
    for k, r in out["extra_runs"].items():
        print(f"{k:10s} wall {r['wall']:.1f}s  {r['s_per_study']:.3f} s/st  steady {r['steady_s_per_study']:.3f}  feed {r['t_feed']:.1f}s")
    if out.get("errors"):
        print("errors:", out["errors"])


if __name__ == "__main__":
    main()
