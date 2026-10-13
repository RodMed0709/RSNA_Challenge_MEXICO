"""exp061 - numeric checks of the knee-centred corpus against the parent corpus (build_corpus96.py) BEFORE training.
Auxiliary script declared in the SPEC (does not train). Exit code 0 = PASS, 1 = FAIL. Writes OUT_DIR/check_kneecrop.json
and OUT_DIR/kneecrop_vs_base.png (6 random studies, middle slice of each slot: parent | knee-centred).

Checks: same N / order of ids.npy; same shape and dtype uint8 (NaN impossible in uint8, still asserted on a float view);
masks identical (tolerance <= 5 slices in the whole corpus, logged); padding slices (mask 0) all zero in every study;
masks recomputed from the volumes on a sample; series that fell back to the matrix centre are byte-identical to the parent
(sample); shifted series differ from the parent; fallback fraction <= 5 %; mean/p50 |shift| mm per slot vs diagnostics
(diag D-2, 300 studies, raw centroid |dx| p50: sagT2 7.0, sagT1 8.0, corT2 2.8, corT1 2.8, ax 2.7, ALL 4.0 mm).
usage: python check_corpus_kneecrop.py --new /root/corpus_kneecrop --base /root/corpus96_384 --out_dir DIR [--sample 300]
"""
import argparse, json, os, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

SLOT_SIZES = [26, 22, 18, 12, 18]
SLOTN = ["sagT2", "sagT1", "corT2", "corT1", "ax"]
B = np.cumsum([0] + SLOT_SIZES)
DIAG_DX_P50 = {"sagT2": 7.0, "sagT1": 8.0, "corT2": 2.8, "corT1": 2.8, "ax": 2.7, "ALL": 4.0}
_G = {}


def _init(new, base):
    _G["n"] = np.load(os.path.join(new, "vols.npy"), mmap_mode="r")
    _G["b"] = np.load(os.path.join(base, "vols.npy"), mmap_mode="r")
    _G["m"] = np.load(os.path.join(new, "masks.npy"))


def _pad(row):  # padding slices of the new corpus are all zero
    z = np.where(_G["m"][row] == 0)[0]
    return int(sum(int(_G["n"][row, d].max()) > 0 for d in z))


def _full(job):  # recompute mask + per-slot identity vs parent
    row, slots_same, slots_shift = job
    v = np.asarray(_G["n"][row]); bv = np.asarray(_G["b"][row])
    mk = (v.reshape(v.shape[0], -1).sum(1) > 0).astype(np.uint8)
    nan = bool(np.isnan(v[:1].astype(np.float32)).any())
    same_bad = sum(int(not np.array_equal(v[B[s]:B[s + 1]], bv[B[s]:B[s + 1]])) for s in slots_same)
    shift_eq = sum(int(np.array_equal(v[B[s]:B[s + 1]], bv[B[s]:B[s + 1]])) for s in slots_shift)
    return row, int((mk != _G["m"][row]).sum()), nan, same_bad, shift_eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="/root/corpus_kneecrop")
    ap.add_argument("--base", default="/root/corpus96_384")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res, fails = {}, []
    ni, bi = np.load(f"{a.new}/ids.npy").astype(str), np.load(f"{a.base}/ids.npy").astype(str)
    res["n_new"], res["n_base"] = len(ni), len(bi)
    res["ids_identical_order"] = bool(len(ni) == len(bi) and (ni == bi).all())
    if not res["ids_identical_order"]: fails.append("ids differ")
    vn = np.load(f"{a.new}/vols.npy", mmap_mode="r"); vb = np.load(f"{a.base}/vols.npy", mmap_mode="r")
    res["shape_new"], res["shape_base"], res["dtype"] = list(vn.shape), list(vb.shape), str(vn.dtype)
    if vn.shape != vb.shape or vn.dtype != np.uint8: fails.append("shape/dtype")
    mn, mb = np.load(f"{a.new}/masks.npy"), np.load(f"{a.base}/masks.npy")
    res["mask_mismatch_slices"] = int((mn != mb).sum()); res["mask_mismatch_studies"] = int((mn != mb).any(1).sum())
    res["valid_slices_mean_new"], res["valid_slices_mean_base"] = float(mn.sum(1).mean()), float(mb.sum(1).mean())
    if res["mask_mismatch_slices"] > 5: fails.append(f"masks differ ({res['mask_mismatch_slices']} slices)")
    ci = pd.read_csv(f"{a.new}/kneecrop_centers.csv")
    meta = json.load(open(f"{a.new}/build_meta.json"))
    res["build_errors"] = len(meta.get("errors", {}))
    res["n_series"] = int(len(ci)); res["fallback_frac"] = float(ci.fallback.mean())
    res["fallback_reasons"] = ci.reason.value_counts().to_dict()
    if res["fallback_frac"] > 0.05: fails.append(f"fallback {res['fallback_frac']:.3f} > 5 %")
    rows = []
    for k, g in list(ci.groupby("slot", sort=False)) + [("ALL", ci)]:
        ok = g[g.fallback == 0]
        rows.append({"slot": k, "n": len(g), "fallback_%": round(100 * g.fallback.mean(), 2),
                     "raw|dx|p50": round(ok.raw_dx_mm.abs().median(), 2), "raw|dx|p90": round(ok.raw_dx_mm.abs().quantile(.9), 2),
                     "raw|dy|p50": round(ok.raw_dy_mm.abs().median(), 2),
                     "eff|dx|mean": round(g.eff_dx_mm.abs().mean(), 2), "eff|dx|p50": round(g.eff_dx_mm.abs().median(), 2),
                     "eff|dy|mean": round(g.eff_dy_mm.abs().mean(), 2), "eff|dy|p50": round(g.eff_dy_mm.abs().median(), 2),
                     "diag_raw|dx|p50": DIAG_DX_P50.get(k), "ms_mean": round(g.ms.mean(), 1)})
    res["shift_table"] = rows
    allr = rows[-1]
    if not (2.0 <= allr["raw|dx|p50"] <= 7.0): fails.append(f"ALL raw |dx| p50 {allr['raw|dx|p50']} outside [2, 7] mm (diag 4.0)")
    res["ms_per_study_1thread"] = meta.get("prepro", {}).get("ms_per_study_1thread")
    # padding zero (all studies) + full sample
    with Pool(a.procs, initializer=_init, initargs=(a.new, a.base)) as p:
        pad_bad = sum(p.map(_pad, range(len(ni)), chunksize=32))
        res["padding_nonzero_slices"] = int(pad_bad)
        if pad_bad: fails.append(f"padding slices non-zero: {pad_bad}")
        rng = np.random.default_rng(0); samp = sorted(rng.choice(len(ni), min(a.sample, len(ni)), replace=False).tolist())
        jobs = []
        for r in samp:
            g = ci[ci.row == r]
            same = [SLOTN.index(s) for s, fb, ey, ex in zip(g.slot, g.fallback, g.eff_dy_mm, g.eff_dx_mm) if fb == 1]
            shift = [SLOTN.index(s) for s, fb, ey, ex in zip(g.slot, g.fallback, g.eff_dy_mm, g.eff_dx_mm)
                     if fb == 0 and max(abs(ey), abs(ex)) > 2 * g.ps.max()]
            jobs.append((r, same, shift))
        out = p.map(_full, jobs, chunksize=4)
    res["sample_studies"] = len(samp)
    res["sample_mask_recompute_mismatch"] = int(sum(o[1] for o in out))
    res["sample_nan"] = int(sum(o[2] for o in out))
    res["sample_fallback_series_checked"] = int(sum(len(j[1]) for j in jobs))
    res["sample_fallback_series_not_identical"] = int(sum(o[3] for o in out))
    res["sample_shifted_series_checked"] = int(sum(len(j[2]) for j in jobs))
    res["sample_shifted_series_identical"] = int(sum(o[4] for o in out))
    if res["sample_mask_recompute_mismatch"]: fails.append("mask != volume content")
    if res["sample_nan"]: fails.append("NaN")
    if res["sample_fallback_series_not_identical"]: fails.append("fallback series differ from parent")
    if res["sample_shifted_series_identical"] > 0.01 * max(1, res["sample_shifted_series_checked"]):
        fails.append("shifted series identical to parent (crop centre not applied?)")
    # PNG
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pick = sorted(np.random.default_rng(1).choice(len(ni), 6, replace=False).tolist())
        fig, ax = plt.subplots(6, 10, figsize=(20, 12.5))
        for i, r in enumerate(pick):
            g = ci[ci.row == r].set_index("slot")
            for s in range(5):
                d = (B[s] + B[s + 1]) // 2
                for j, (v, lab) in enumerate(((vb, "parent"), (vn, "knee"))):
                    x = ax[i, 2 * s + j]; x.imshow(v[r, d], cmap="gray", vmin=0, vmax=255); x.set_xticks([]); x.set_yticks([])
                    t = f"{SLOTN[s]} {lab}"
                    if j == 1 and SLOTN[s] in g.index:
                        q = g.loc[SLOTN[s]]; t += f" dy{q.eff_dy_mm:+.0f} dx{q.eff_dx_mm:+.0f}{' FB' if q.fallback else ''}"
                    x.set_title(t, fontsize=7)
            ax[i, 0].set_ylabel(f"row {r}", fontsize=7)
        fig.suptitle("exp061: parent centre crop vs knee-centred crop (middle slice per slot; eff shift in mm)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "kneecrop_vs_base.png"), dpi=80); plt.close(fig)
        res["png"] = os.path.join(a.out_dir, "kneecrop_vs_base.png"); res["png_rows"] = pick
    except Exception as e:
        res["png_error"] = f"{type(e).__name__}: {e}"
    res["fails"] = fails; res["status"] = "PASS" if not fails else "FAIL"
    json.dump(res, open(os.path.join(a.out_dir, "check_kneecrop.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).set_index("slot").to_string())
    print(json.dumps({k: v for k, v in res.items() if k != "shift_table"}, default=str))
    print("CHECK", res["status"], fails, flush=True)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
