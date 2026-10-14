"""N4 bias field correction (Tustison et al. 2010, IEEE TMI) of the Raptor d96/384 corpus. Auxiliary script (does not train).

Per study and per series slot (5 slots: Sag fluid 26, Sag nonfluid 22, Cor fluid 18, Cor 12, Ax 18) the valid slices of
the slot form a 3D stack (z, 384, 384). On float(uint8) + 1 we run SimpleITK N4BiasFieldCorrectionImageFilter with an Otsu
mask (sitk.OtsuThreshold 0/1), shrink factor 4 in x, y and z (z only if the slot has >= 8 valid slices), default iterations
and control points. The log bias field is evaluated at full resolution and divided out; the corrected stack is then
re-normalised with its own 2-98 percentile (over all voxels of the slot stack, as build_corpus96.py does per series)
-> uint8. Invalid slices stay zero. If N4 fails on a slot, the slot is kept unchanged and counted in errors.

Input:  SRC/vols.npy (N, 96, 384, 384) uint8, masks.npy, ids.npy, build_meta.json (build_corpus96.py).
Output: OUT/vols.npy (same shape), masks/ids copied, build_meta.json = SRC meta + {"prepro": {...}}.
Inference (Kaggle) must apply the same function `n4_study(vol, mask)` to each study before cropping / windows.

Usage: python build_corpus_n4.py --src /root/corpus96_384 --out /root/corpus_n4 --procs 16 [--limit 40]
"""
import argparse, json, os, shutil, time
from multiprocessing import Pool

import numpy as np

SLOTS = [26, 22, 18, 12, 18]
BOUNDS = np.cumsum([0] + SLOTS)
SHRINK = 4
_G = {}


def n4_stack(stack):
    """stack: (z, H, W) uint8 -> (z, H, W) uint8 after N4 + 2-98 percentile."""
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(stack.astype(np.float32) + 1.0)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    # SimpleITK order (x, y, z); z shrunk too (ANTs-style isotropic factor) unless the slot is too thin
    shrink = [SHRINK, SHRINK] + ([SHRINK if stack.shape[0] >= 2 * SHRINK else 1] if img.GetDimension() == 3 else [])
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(sitk.Shrink(img, shrink), sitk.Shrink(mask, shrink))
    log_bias = sitk.GetArrayFromImage(corrector.GetLogBiasFieldAsImage(img))
    corr = (stack.astype(np.float32) + 1.0) / np.exp(log_bias)
    lo, hi = np.percentile(corr, [2.0, 98.0])
    return (np.clip((corr - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def n4_study(vol, mask):
    out = np.zeros_like(vol); errs = 0
    for s in range(len(SLOTS)):
        d = np.arange(BOUNDS[s], BOUNDS[s + 1])
        d = d[mask[d] > 0]
        if len(d) == 0:
            continue
        stack = vol[d]
        try:
            out[d] = n4_stack(stack if len(d) > 1 else stack[0])  # 1 slice -> 2D N4
        except Exception:
            out[d] = stack; errs += 1
    return out, errs


def _init(src, out):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    _G["src"] = np.load(os.path.join(src, "vols.npy"), mmap_mode="r")
    _G["dst"] = np.load(os.path.join(out, "vols.npy"), mmap_mode="r+")
    _G["masks"] = np.load(os.path.join(src, "masks.npy"))


def _one(row):
    vol = np.array(_G["src"][row])
    t = time.time()
    out, errs = n4_study(vol, _G["masks"][row])
    dt = time.time() - t
    _G["dst"][row] = out
    return row, dt, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/corpus96_384")
    ap.add_argument("--out", default="/root/corpus_n4")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = np.load(os.path.join(a.src, "vols.npy"), mmap_mode="r")
    N = a.limit or src.shape[0]
    dst = np.lib.format.open_memmap(os.path.join(a.out, "vols.npy"), mode="w+", dtype=np.uint8, shape=src.shape)
    del dst, src
    t0 = time.time(); done = 0; ts = []; nerr = 0
    with Pool(a.procs, initializer=_init, initargs=(a.src, a.out)) as pool:
        for row, dt, errs in pool.imap_unordered(_one, range(N), chunksize=2):
            ts.append(dt); done += 1; nerr += errs
            if done % 100 == 0 or done == N:
                el = time.time() - t0
                print(f"{done}/{N} | {el:.0f}s | {done/el:.2f} est/s | ETA {(N-done)/(done/el)/60:.1f} min "
                      f"| N4 {np.mean(ts):.2f} s/estudio (1 hilo) | slots con error {nerr}", flush=True)
    for f in ("masks.npy", "ids.npy"):
        shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    meta = json.load(open(os.path.join(a.src, "build_meta.json")))
    meta["prepro"] = {"method": "n4", "ref": "Tustison et al. 2010 IEEE TMI", "lib": "SimpleITK N4BiasFieldCorrectionImageFilter",
                      "mask": "OtsuThreshold", "shrink": [SHRINK, SHRINK, "4 if >=8 slices else 1"], "iterations": "default", "renorm": "per slot 2-98 pct",
                      "src": a.src, "n": N, "procs": a.procs, "build_seconds": time.time() - t0, "slot_errors": nerr,
                      "ms_per_study_1thread": 1000 * float(np.mean(ts)), "ms_per_study_p95": 1000 * float(np.percentile(ts, 95))}
    json.dump(meta, open(os.path.join(a.out, "build_meta.json"), "w"), indent=1)
    print("DONE", json.dumps(meta["prepro"]), flush=True)


if __name__ == "__main__":
    main()
