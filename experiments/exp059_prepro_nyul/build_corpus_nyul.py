"""Nyul-Udupa histogram standardization (Nyul et al. 2000, IEEE TMI; same algorithm as TorchIO HistogramStandardization)
of the Raptor d96/384 corpus. Auxiliary script (does not train).

Landmarks: percentiles PCTS = [1, 10, 20, ..., 90, 99] of the foreground voxels (voxel > mean of the slot stack, TorchIO's
default mask) of each series slot stack (valid slices of the slot). Standard scale: for every TRAIN study of fold 0
(split_v1_seed42, fold not in {0, -1, -2}: no fold-0 validation, no gold) the landmarks are mapped linearly so that
p1 -> 0 and p99 -> 255, and averaged per slot type (5 slot types) -> standard landmarks (5, 11).
Apply: per study and slot, piecewise-linear map own landmarks -> standard landmarks (edge segments extrapolate, as in
TorchIO), as a 256-entry LUT on the uint8 values; clip to [0, 255] -> uint8. Invalid slices stay zero.

Output: OUT/vols.npy, masks/ids copied, OUT/study_landmarks.npy (N, 5, 11), build_meta.json + {"prepro": {...}},
        and LANDMARKS json (standard scale; needed by the Kaggle inference kernel).
Usage: python build_corpus_nyul.py --src /root/corpus96_384 --out /root/corpus_nyul --landmarks <json> --procs 16
"""
import argparse, json, os, shutil, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SLOTS = [26, 22, 18, 12, 18]
SLOT_NAMES = ["sag_fluid", "sag_nonfluid", "cor_fluid", "cor", "ax"]
BOUNDS = np.cumsum([0] + SLOTS)
PCTS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
S_MIN, S_MAX = 0.0, 255.0
_G = {}


def slot_slices(mask, s):
    d = np.arange(BOUNDS[s], BOUNDS[s + 1])
    return d[mask[d] > 0]


def landmarks_stack(stack):
    x = stack.astype(np.float32)
    fg = x[x > x.mean()]
    return np.percentile(fg if fg.size else x, PCTS)


def lut_from(own, std):
    """TorchIO _normalize as a LUT over 0..255: piecewise linear own -> std, edge segments extrapolated."""
    own = np.asarray(own, np.float64); std = np.asarray(std, np.float64)
    d_own = np.diff(own); d_std = np.diff(std)
    d_own[d_own == 0] = np.inf
    slope = d_std / d_own; icpt = std[:-1] - slope * own[:-1]
    v = np.arange(256, dtype=np.float64)
    b = np.digitize(v, own[1:-1], right=False)
    return np.clip(np.rint(slope[b] * v + icpt[b]), 0, 255).astype(np.uint8)


def nyul_study(vol, mask, std):
    out = np.zeros_like(vol)
    for s in range(len(SLOTS)):
        d = slot_slices(mask, s)
        if len(d):
            out[d] = lut_from(landmarks_stack(vol[d]), std[s])[vol[d]]
    return out


def _init(src, out, std):
    _G["src"] = np.load(os.path.join(src, "vols.npy"), mmap_mode="r")
    _G["dst"] = np.load(os.path.join(out, "vols.npy"), mmap_mode="r+") if out else None
    _G["masks"] = np.load(os.path.join(src, "masks.npy"))
    _G["std"] = std


def _lm(row):
    vol = np.array(_G["src"][row]); m = _G["masks"][row]
    lm = np.full((len(SLOTS), len(PCTS)), np.nan)
    for s in range(len(SLOTS)):
        d = slot_slices(m, s)
        if len(d):
            lm[s] = landmarks_stack(vol[d])
    return row, lm


def _apply(row):
    vol = np.array(_G["src"][row])
    t = time.time()
    out = nyul_study(vol, _G["masks"][row], _G["std"])  # landmarks + LUT, the full inference-time cost
    dt = time.time() - t
    _G["dst"][row] = out
    return row, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/corpus96_384")
    ap.add_argument("--out", default="/root/corpus_nyul")
    ap.add_argument("--landmarks", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve()))
    from src.knee.splits import load_split
    split = load_split("split_v1_seed42")
    train = set(split.loc[(~split.fold.isin([0, -1, -2])) & (split.is_gold == 0), "StudyInstanceUID"].astype(str))
    ids = np.load(os.path.join(a.src, "ids.npy")).astype(str)
    src = np.load(os.path.join(a.src, "vols.npy"), mmap_mode="r")
    N = a.limit or src.shape[0]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    # pass 1: per-study landmarks
    lms = np.full((src.shape[0], len(SLOTS), len(PCTS)), np.nan)
    with Pool(a.procs, initializer=_init, initargs=(a.src, None, None)) as pool:
        for row, lm in pool.imap_unordered(_lm, range(N), chunksize=8):
            lms[row] = lm
    t_lm = time.time() - t0
    tr_rows = np.array([i for i in range(N) if ids[i] in train])
    std = np.zeros((len(SLOTS), len(PCTS))); n_used = []
    for s in range(len(SLOTS)):
        L = lms[tr_rows, s]; L = L[~np.isnan(L).any(1) & (L[:, -1] > L[:, 0])]
        mapped = (L - L[:, :1]) / (L[:, -1:] - L[:, :1]) * (S_MAX - S_MIN) + S_MIN
        std[s] = mapped.mean(0); n_used.append(int(len(L)))
    print(f"landmarks: {t_lm:.0f}s | train studies {len(tr_rows)} | per slot {n_used}", flush=True)
    for s in range(len(SLOTS)):
        print(SLOT_NAMES[s], np.round(std[s], 1).tolist(), flush=True)
    lm_json = {"method": "nyul_udupa_2000", "percentiles": PCTS, "scale": [S_MIN, S_MAX], "mask": "voxel > mean of slot stack",
               "slots": SLOT_NAMES, "slot_sizes": SLOTS, "standard_landmarks": std.tolist(), "n_train_studies_per_slot": n_used,
               "fit_on": "split_v1_seed42 fold-0 TRAIN only (fold not in {0,-1,-2}, is_gold == 0)", "src": a.src}
    Path(a.landmarks).parent.mkdir(parents=True, exist_ok=True)
    json.dump(lm_json, open(a.landmarks, "w"), indent=1)
    np.save(os.path.join(a.out, "study_landmarks.npy"), lms)
    # pass 2: apply
    dst = np.lib.format.open_memmap(os.path.join(a.out, "vols.npy"), mode="w+", dtype=np.uint8, shape=src.shape)
    del dst, src
    ts = []; done = 0
    with Pool(a.procs, initializer=_init, initargs=(a.src, a.out, std)) as pool:
        for row, dt in pool.imap_unordered(_apply, range(N), chunksize=4):
            ts.append(dt); done += 1
            if done % 500 == 0 or done == N:
                el = time.time() - t0
                print(f"apply {done}/{N} | {el:.0f}s | {1000*np.mean(ts):.1f} ms/estudio (1 hilo)", flush=True)
    for f in ("masks.npy", "ids.npy"):
        shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    meta = json.load(open(os.path.join(a.src, "build_meta.json")))
    meta["prepro"] = {"method": "nyul", "ref": "Nyul, Udupa, Zhang 2000 IEEE TMI", "landmarks_file": a.landmarks,
                      "percentiles": PCTS, "scale": [S_MIN, S_MAX], "n_train": n_used, "src": a.src, "n": N,
                      "procs": a.procs, "build_seconds": time.time() - t0, "landmarks_seconds": t_lm,
                      "ms_per_study_1thread": 1000 * float(np.mean(ts)), "ms_per_study_p95": 1000 * float(np.percentile(ts, 95))}
    json.dump(meta, open(os.path.join(a.out, "build_meta.json"), "w"), indent=1)
    print("DONE", json.dumps(meta["prepro"]), flush=True)


if __name__ == "__main__":
    main()
