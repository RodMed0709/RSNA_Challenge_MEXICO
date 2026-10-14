"""exp032 - tabla de alineación izquierda/derecha para el corpus Raptor d96 (sin reconstruirlo).

Script auxiliar del experimento (no entrena). Para cada estudio del corpus `raptor96_384` reproduce la elección de
series de `build_corpus96.py` (`_pick_series_for_slot`, mismo orden de train_series.csv), lee la cabecera DICOM
(IOP) de la primera imagen de cada serie elegida y decide qué eje del bloque de cada slot lleva el eje x del
paciente (L/R) y en qué sentido:
  eje 0 = orden de cortes (build_corpus96 ordena ascendente por IPP·(r×c), dirección n = r×c)
  eje 1 = filas de la imagen (dirección c = IOP[3:]),  eje 2 = columnas (dirección r = IOP[:3])
El lado del estudio sale de `study_meta_v2.csv` (`study_side`, cascada tag → texto → geometría de exp009).

Convención: m = signo_x(eje) · (+1 si R, −1 si L); m = +1 ⇔ el índice crece hacia medial. Objetivo por slot =
el m mayoritario de las rodillas derechas (las derechas típicas no se tocan). Se voltea el eje x de un slot si
el estudio tiene lado conocido y su m ≠ objetivo. Lado `unknown` o serie sin IOP → nativo (no se toca).

Salida: <out>.csv (estudio × slot) y <out>.npy int8 (N, 5) alineado con ids.npy:
  0 = nativo · 1 = invertir orden de cortes del slot · 2 = voltear filas · 3 = voltear columnas
Uso: python lateralidad_corpus.py --corpus /root/corpus96_384 --out /root/corpus96_384/align_lr_exp032
"""
import argparse, glob, json, os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

RAW = "/workspace/data/raw"
META = "/workspace/data/cache/meta_v2/study_meta_v2.csv"
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]


def _pick_series_for_slot(rows, plane, fluid, used):  # verbatim de build_corpus96.py
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def _iop(sid, ser):
    import pydicom
    fs = sorted(glob.glob(f"{RAW}/train_series/{sid}/{ser}/*.dcm"))
    for f in fs[:3]:
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            v = getattr(h, "ImageOrientationPatient", None)
            if v is not None and len(v) == 6:
                return np.array(v, float)
        except Exception:
            pass
    return None


def x_axis_sign(iop):
    r, c = iop[:3], iop[3:]
    n = np.cross(r, c)
    dirs = [n, c, r]                       # eje 0, 1, 2
    ax = int(np.argmax([abs(d[0]) for d in dirs]))
    return ax, int(np.sign(dirs[ax][0]) or 1), float(abs(dirs[ax][0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/root/corpus96_384")
    ap.add_argument("--out", default="/root/corpus96_384/align_lr_exp032")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    ids = np.load(f"{a.corpus}/ids.npy").astype(str)
    ser = pd.read_csv(f"{RAW}/train_series.csv")
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str)
    ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    recs = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    side = pd.read_csv(META).set_index("study")
    jobs = []
    for row, sid in enumerate(ids):
        used = set()
        for s, (plane, fluid, k) in enumerate(SLOTS):
            r = _pick_series_for_slot(recs.get(sid, []), plane, fluid, used)
            if r is None:
                continue
            used.add(r["SeriesInstanceUID"])
            jobs.append((row, sid, s, plane, r["SeriesInstanceUID"]))
    with ThreadPoolExecutor(a.threads) as ex:
        iops = list(ex.map(lambda j: _iop(j[1], j[4]), jobs))
    out = []
    for (row, sid, s, plane, uid), iop in zip(jobs, iops):
        st = side.loc[sid, "study_side"] if sid in side.index else "unknown"
        src = side.loc[sid, "study_side_src"] if sid in side.index else "unknown"
        if iop is None:
            out.append(dict(row=row, study=sid, slot=s, plane=plane, series=uid, x_axis=-1, x_sign=0, x_cos=np.nan,
                            side=st, side_src=src))
            continue
        ax, sg, cos = x_axis_sign(iop)
        out.append(dict(row=row, study=sid, slot=s, plane=plane, series=uid, x_axis=ax, x_sign=sg, x_cos=cos,
                        side=st, side_src=src))
    df = pd.DataFrame(out)
    df["m"] = df.x_sign * df.side.map({"R": 1, "L": -1}).fillna(0).astype(int)
    target = {}
    for s in range(len(SLOTS)):
        rr = df[(df.slot == s) & (df.side == "R") & (df.x_axis >= 0)]
        ax_t = int(rr.x_axis.mode()[0]); m_t = int(rr.m.mode()[0])
        target[s] = {"x_axis": ax_t, "m": m_t,
                     "R_axis_share": float((rr.x_axis == ax_t).mean()), "R_m_share": float((rr.m == m_t).mean())}
    df["flip"] = (df.m != 0) & (df.x_axis >= 0) & (df.m != df.slot.map(lambda s: target[s]["m"]))
    df["code"] = np.where(df.flip, df.x_axis + 1, 0)
    codes = np.zeros((len(ids), len(SLOTS)), np.int8)
    codes[df.row.to_numpy(), df.slot.to_numpy()] = df.code.to_numpy()
    df.to_csv(a.out + ".csv", index=False)
    np.save(a.out + ".npy", codes)
    st_side = pd.Series([side.loc[u, "study_side"] if u in side.index else "unknown" for u in ids])
    summ = {
        "n_studies": len(ids), "n_series": len(df), "series_without_iop": int((df.x_axis < 0).sum()),
        "study_side_counts": st_side.value_counts().to_dict(),
        "study_side_src_counts": pd.Series([side.loc[u, "study_side_src"] if u in side.index else "unknown"
                                            for u in ids]).value_counts().to_dict(),
        "target_per_slot": target,
        "flips_per_slot": {int(s): int(g.flip.sum()) for s, g in df.groupby("slot")},
        "flip_by_side_per_slot": {f"{s}_{sd}": float(g.flip.mean()) for (s, sd), g in df.groupby(["slot", "side"])},
        "x_axis_counts_per_slot": {int(s): g.x_axis.value_counts().to_dict() for s, g in df.groupby("slot")},
        "x_cos_min": float(df.x_cos.min()), "x_cos_lt_0.7": int((df.x_cos < 0.7).sum()),
        "studies_with_any_flip": int((codes != 0).any(1).sum()),
    }
    json.dump(summ, open(a.out + ".json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
