"""exp051 - corpus denso Raptor/nartaa d96 (96 cortes x 384 px, uint8) desde los DICOM de train, con
normalizacion por percentiles PCT (0.5-99.5) de los cortes elegidos de cada serie en vez de 2-98.
Copia de experiments/exp029_raptor_entreno/build_corpus96.py; unico cambio: PCT (y "pct" en build_meta).

Script auxiliar del experimento (no es codigo de entrenamiento): porta `build_study` del notebook publico
`nartaa/rsna-knee-0949-anatomical-mirror` (preprocesado verbatim de kprep2/dino_preprocess.py) y lo aplica
a los 4407 estudios de train en paralelo. Salida en OUT (disco local del pod):
  vols.npy  (N, 96, 384, 384) uint8   (np.lib.format.open_memmap, se escribe por filas desde cada proceso)
  masks.npy (N, 96) uint8             (corte valido = imagen no toda a cero)
  ids.npy   (N,) str                  (orden de train.csv)
  build_meta.json

Uso:  python build_corpus_norm.py --out /root/corpus96_384_p05 --procs 32 [--limit 40]
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
PCT = (0.5, 99.5)  # exp051 (build_corpus96.py: 2.0, 98.0)
HDR_THREADS = 8
PX_THREADS = 8


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

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
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
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in SLOTS:
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
            loq, hiq = np.percentile(allpx, list(PCT))
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS: break
            if a is None: idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


_G = {}


def _init(out, ser):
    _G["reader"] = _make_reader()
    _G["vols"] = np.load(os.path.join(out, "vols.npy"), mmap_mode="r+")
    _G["ser"] = ser


def _one(job):
    row, sid = job
    t = time.time()
    try:
        vol, mask = build_study(sid, _G["ser"], f"{RAW}/train_series", _G["reader"])
        err = None
    except Exception as e:
        vol = np.zeros((MAXS, IMG, IMG), np.uint8); mask = np.zeros(MAXS, np.uint8)
        err = f"{type(e).__name__}: {e}"
    _G["vols"][row] = vol
    return row, mask, err, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/corpus96_384_p05")
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
    masks = np.zeros((N, MAXS), np.uint8); errs = {}
    t0 = time.time(); done = 0
    with Pool(a.procs, initializer=_init, initargs=(a.out, ser)) as pool:
        for row, mask, err, dt in pool.imap_unordered(_one, list(enumerate(ids)), chunksize=1):
            masks[row] = mask; done += 1
            if err: errs[ids[row]] = err; print(f"[err] {ids[row]} {err}", flush=True)
            if done % 50 == 0 or done == N:
                el = time.time() - t0
                print(f"{done}/{N} | {el:.0f}s | {done/el:.2f} est/s | ETA {(N-done)/(done/el)/60:.1f} min "
                      f"| last {dt:.1f}s", flush=True)
    np.save(os.path.join(a.out, "masks.npy"), masks)
    np.save(os.path.join(a.out, "ids.npy"), np.array(ids))
    meta = {"n": N, "pct": PCT, "slots": SLOTS, "span": SPAN, "img": IMG, "crop_mm": CROP_MM, "procs": a.procs,
            "seconds": time.time() - t0, "errors": errs,
            "valid_slices_mean": float(masks.sum(1).mean()),
            "empty_studies": int((masks.sum(1) == 0).sum()),
            "source": "build_study from nartaa/rsna-knee-0949-anatomical-mirror (verbatim) with percentiles PCT"}
    json.dump(meta, open(os.path.join(a.out, "build_meta.json"), "w"), indent=1)
    print("DONE", json.dumps({k: v for k, v in meta.items() if k != "errors"}), "errors", len(errs), flush=True)


if __name__ == "__main__":
    main()
