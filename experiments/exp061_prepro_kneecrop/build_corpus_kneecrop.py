"""exp061 (copy of exp029/build_corpus96.py) - knee-centred 140 mm crop. ONLY change vs build_corpus96.py: the centre of
the 140 mm crop is the knee-tissue centre of each SERIES (one centre for all its slices) instead of the matrix centre.
Centre = centroid of the largest connected component of the Otsu foreground mask of the max-projection of the chosen,
intensity-windowed slices (same as orq/p7c_prepro/diag/d_px.py, docs/conocimiento/09_preprocesado.md P-1); shift clamped
to +-25 mm per axis; the crop window is then clamped to stay inside the image (no padding, like the parent).
Fallback to the matrix centre (byte-identical to the parent) if the mask fails (area < 2 % of the image, or error).
Origin: CoPAS (Qiu 2024, doi:10.1038/s41467-024-51888-4); Xie 2025 (doi:10.1016/j.eclinm.2025.103534).
Extra outputs: OUT/kneecrop_centers.csv (per series: slot, fallback, raw/effective shift mm, centring ms) and
build_meta["prepro"] (ms_per_study_1thread = centring cost only, measured with cv2.setNumThreads(1)).
Usage: python build_corpus_kneecrop.py --out /root/corpus_kneecrop --procs 22 [--limit 40]

--- original docstring ---
exp029 - corpus denso Raptor/nartaa d96 (96 cortes x 384 px, uint8) desde los DICOM de train.

Script auxiliar del experimento (no es codigo de entrenamiento): porta `build_study` del notebook publico
`nartaa/rsna-knee-0949-anatomical-mirror` (preprocesado verbatim de kprep2/dino_preprocess.py) y lo aplica
a los 4407 estudios de train en paralelo. Salida en OUT (disco local del pod):
  vols.npy  (N, 96, 384, 384) uint8   (np.lib.format.open_memmap, se escribe por filas desde cada proceso)
  masks.npy (N, 96) uint8             (corte valido = imagen no toda a cero)
  ids.npy   (N,) str                  (orden de train.csv)
  build_meta.json

Uso:  python build_corpus96.py --out /root/corpus96_384 --procs 32 [--limit 40]
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd

RAW = "/workspace/data/raw"
IMG = 384
CROP_MM = 140.0
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(s[2] for s in SLOTS)  # 96
SPAN = (0.02, 0.98)
HDR_THREADS = 8
PX_THREADS = 8
MAX_SHIFT_MM = 25.0
MIN_AREA_FRAC = 0.02
SLOTN = ["sagT2", "sagT1", "corT2", "corT1", "ax"]


def fg_mask(proj):
    """proj in [0,1] -> largest connected foreground component (Otsu on 8-bit, closing/opening). = diag d_px.py"""
    import cv2
    u8 = (proj * 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (5, 5), 0)
    _, m = cv2.threshold(u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return m.astype(bool)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab == k


def knee_center(W, ps):
    """W: windowed slices in [0,1] (native px). Returns (cy, cx, ok, reason, raw_dy_mm, raw_dx_mm) in the px frame of the
    middle slice; (h/2, w/2, False, ...) on failure."""
    import cv2
    shp = W[len(W) // 2].shape; h, w = shp
    try:
        S = [a if a.shape == shp else cv2.resize(a, shp[::-1]) for a in W]
        m = fg_mask(np.max(np.stack(S), 0))
        ys, xs = np.nonzero(m)
        if len(ys) < MIN_AREA_FRAC * h * w:
            return h / 2, w / 2, False, "small_mask", 0.0, 0.0
        cy, cx = float(ys.mean()), float(xs.mean())
        rdy, rdx = (cy - h / 2) * ps, (cx - w / 2) * ps
        lim = MAX_SHIFT_MM / max(ps, 1e-3)
        return (float(np.clip(cy, h / 2 - lim, h / 2 + lim)), float(np.clip(cx, w / 2 - lim, w / 2 + lim)), True,
                "ok", rdy, rdx)
    except Exception as e:
        return h / 2, w / 2, False, f"err:{type(e).__name__}", 0.0, 0.0


def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def _hdr(f):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, 'ImageOrientationPatient', None)
            ipp = getattr(h, 'ImagePositionPatient', None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, 'InstanceNumber', 0) or 0)
            ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
            return (pos, f, ps, True)
        except Exception:
            return (0.0, f, 0.5, False)

    def order_and_meta(sdir):
        fs = glob.glob(sdir + "/*.dcm")
        with ThreadPoolExecutor(max_workers=HDR_THREADS) as ex:
            out = list(ex.map(_hdr, fs))
        recs = [(pos, f, ps) for pos, f, ps, _ in out]; ps_list = [ps for _, _, ps, ok in out if ok]
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps, c=None):
        """c = (cy_frac, cx_frac) knee centre as a fraction of the slice size; None -> matrix centre (parent)."""
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        if c is not None:
            y0 = int(np.clip(int(round(c[0] * h - cpx / 2)), 0, h - cpx))
            x0 = int(np.clip(int(round(c[1] * w - cpx / 2)), 0, w - cpx))
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set(); info = []
    for si, (plane, fluid, k) in enumerate(SLOTS):
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        n = len(files); lo, hi = int(n * SPAN[0]), int(n * SPAN[1]) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        sel = [files[min(p, n - 1)] for p in picks]

        def _px(fp_ps):
            fp, ps = fp_ps
            try: return read_px(fp), ps
            except Exception: return None, med_ps
        with ThreadPoolExecutor(max_workers=PX_THREADS) as ex:
            got = list(ex.map(_px, sel))
        arrs = [a for a, _ in got]; pss = [ps for _, ps in got]
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        AW = [None if a is None else np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1) for a in arrs]
        c = None
        if valid:   # exp061: one knee centre per series (timed: this is the only extra inference cost)
            t0 = time.perf_counter()
            vw = [x for x in AW if x is not None]; vps = [p for x, p in zip(AW, pss) if x is not None]
            ps_mid = vps[len(vps) // 2] if vps[len(vps) // 2] > 0 else med_ps
            cy, cx, ok, why, rdy, rdx = knee_center(vw, ps_mid)
            H, Wd = vw[len(vw) // 2].shape
            c = (cy / H, cx / Wd) if ok else None
            ms = 1000 * (time.perf_counter() - t0)
            cpx = min(int(round(CROP_MM / max(ps_mid, 1e-3))), min(H, Wd))
            if c is None:
                ey = ex = 0.0
            else:
                y0 = int(np.clip(int(round(c[0] * H - cpx / 2)), 0, H - cpx))
                x0 = int(np.clip(int(round(c[1] * Wd - cpx / 2)), 0, Wd - cpx))
                ey = (y0 + cpx / 2 - H / 2) * ps_mid; ex = (x0 + cpx / 2 - Wd / 2) * ps_mid
            info.append(dict(slot=SLOTN[si], fallback=int(not ok), reason=why, raw_dy_mm=rdy, raw_dx_mm=rdx,
                             eff_dy_mm=ey, eff_dx_mm=ex, h=H, w=Wd, ps=ps_mid, cpx=cpx, ms=ms))
        for aw, ps in zip(AW, pss):
            if idx >= MAXS: break
            if aw is None: idx += 1; continue
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps, c)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask, info


_G = {}


def _init(out, ser):
    import cv2
    cv2.setNumThreads(1)   # exp061: centring cost measured on 1 thread
    _G["reader"] = _make_reader()
    _G["vols"] = np.load(os.path.join(out, "vols.npy"), mmap_mode="r+")
    _G["ser"] = ser


def _one(job):
    row, sid = job
    t = time.time()
    try:
        vol, mask, info = build_study(sid, _G["ser"], f"{RAW}/train_series", _G["reader"])
        err = None
    except Exception as e:
        vol = np.zeros((MAXS, IMG, IMG), np.uint8); mask = np.zeros(MAXS, np.uint8); info = []
        err = f"{type(e).__name__}: {e}"
    _G["vols"][row] = vol
    return row, mask, err, time.time() - t, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/corpus96_384")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke_gold", action="store_true", help="con --limit: gold (58) + los primeros --limit no-gold")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr = pd.read_csv(f"{RAW}/train.csv"); ids = tr["StudyInstanceUID"].astype(str).tolist()
    if a.limit and a.smoke_gold:
        LABS = [c for c in tr.columns if c not in ("StudyInstanceUID",) and tr[c].dtype != object]
        isg = tr[LABS].notna().all(axis=1).to_numpy()
        ids = [u for u, g in zip(ids, isg) if g] + [u for u, g in zip(ids, isg) if not g][:a.limit]
    elif a.limit: ids = ids[:a.limit]
    ser = pd.read_csv(f"{RAW}/train_series.csv")
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str)
    ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    ser = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    N = len(ids)
    vols = np.lib.format.open_memmap(os.path.join(a.out, "vols.npy"), mode="w+", dtype=np.uint8,
                                     shape=(N, MAXS, IMG, IMG))
    del vols
    masks = np.zeros((N, MAXS), np.uint8); errs = {}; infos = []
    t0 = time.time(); done = 0
    with Pool(a.procs, initializer=_init, initargs=(a.out, ser)) as pool:
        for row, mask, err, dt, info in pool.imap_unordered(_one, list(enumerate(ids)), chunksize=1):
            masks[row] = mask; done += 1
            infos += [dict(row=row, sid=ids[row], **d) for d in info]
            if err: errs[ids[row]] = err; print(f"[err] {ids[row]} {err}", flush=True)
            if done % 50 == 0 or done == N:
                el = time.time() - t0
                print(f"{done}/{N} | {el:.0f}s | {done/el:.2f} est/s | ETA {(N-done)/(done/el)/60:.1f} min "
                      f"| last {dt:.1f}s", flush=True)
    np.save(os.path.join(a.out, "masks.npy"), masks)
    np.save(os.path.join(a.out, "ids.npy"), np.array(ids))
    meta = {"n": N, "slots": SLOTS, "span": SPAN, "img": IMG, "crop_mm": CROP_MM, "procs": a.procs,
            "seconds": time.time() - t0, "errors": errs,
            "valid_slices_mean": float(masks.sum(1).mean()),
            "empty_studies": int((masks.sum(1) == 0).sum()),
            "source": "build_study from nartaa/rsna-knee-0949-anatomical-mirror (verbatim) + exp061 knee-centred crop"}
    ci = pd.DataFrame(infos).sort_values(["row", "slot"])
    ci.to_csv(os.path.join(a.out, "kneecrop_centers.csv"), index=False)
    per_study_ms = ci.groupby("row").ms.sum()
    meta["prepro"] = {"method": "kneecrop",
                      "ref": "CoPAS doi:10.1038/s41467-024-51888-4; Xie 2025 doi:10.1016/j.eclinm.2025.103534",
                      "center": "centroid of largest CC of Otsu mask on max-projection of chosen windowed slices (diag d_px.py)",
                      "max_shift_mm": MAX_SHIFT_MM, "min_area_frac": MIN_AREA_FRAC, "crop_mm": CROP_MM, "n": N,
                      "procs": a.procs, "build_seconds": meta["seconds"], "n_series": int(len(ci)),
                      "fallback_frac": float(ci.fallback.mean()),
                      "ms_per_study_1thread": float(per_study_ms.mean()),
                      "ms_per_study_p95": float(per_study_ms.quantile(.95)),
                      "abs_eff_dx_mm_p50": float(ci.eff_dx_mm.abs().median()),
                      "abs_raw_dx_mm_p50": float(ci.raw_dx_mm.abs().median())}
    json.dump(meta, open(os.path.join(a.out, "build_meta.json"), "w"), indent=1)
    print("DONE", json.dumps({k: v for k, v in meta.items() if k != "errors"}), "errors", len(errs), flush=True)


if __name__ == "__main__":
    main()
