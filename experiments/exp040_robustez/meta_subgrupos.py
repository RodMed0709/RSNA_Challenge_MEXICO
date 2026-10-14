"""exp040 - metadatos de subgrupo para los 929 estudios (val fold 0 + gold). Script auxiliar declarado.

Fuentes: fabricante y campo (T) de /workspace/data/cache/meta_v2/series_meta_v2.csv (moda sobre las series del estudio),
idioma del informe y lado de study_meta_v2.csv, sexo y edad de la cabecera DICOM (un fichero por estudio).
Salida: results/exp040_robustez/meta_subgrupos.csv
"""
import glob, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.knee.splits import load_split

RAW = "/workspace/data/raw/train_series"
OUT = REPO / "results" / "exp040_robustez" / "meta_subgrupos.csv"


def hdr(uid):
    fs = sorted(glob.glob(f"{RAW}/{uid}/*/*.dcm"))[:1]
    if not fs:
        return uid, None, None
    try:
        d = pydicom.dcmread(fs[0], stop_before_pixels=True)
        return uid, getattr(d, "PatientSex", None), getattr(d, "PatientAge", None)
    except Exception:
        return uid, None, None


def mode(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None


def main():
    split = load_split("split_v1_seed42")
    sub = split[(split.fold == 0) | (split.is_gold == 1)].copy()
    sub["StudyInstanceUID"] = sub.StudyInstanceUID.astype(str)
    ids = list(sub.StudyInstanceUID)
    ser = pd.read_csv("/workspace/data/cache/meta_v2/series_meta_v2.csv", usecols=["study", "manufacturer", "field_t"])
    ser = ser[ser.study.isin(ids)]
    g = ser.groupby("study").agg(manufacturer=("manufacturer", mode), field_t=("field_t", mode))
    st = pd.read_csv("/workspace/data/cache/meta_v2/study_meta_v2.csv", usecols=["study", "lang", "study_side"]).set_index("study")
    with ThreadPoolExecutor(16) as ex:
        h = pd.DataFrame(list(ex.map(hdr, ids)), columns=["study", "sex", "age"]).set_index("study")
    out = pd.DataFrame(index=pd.Index(ids, name="StudyInstanceUID"))
    out["set"] = ["gold" if x == 1 else "val" for x in sub.set_index("StudyInstanceUID").loc[ids, "is_gold"]]
    out = out.join(g).join(st).join(h)
    out.to_csv(OUT)
    for c in ["manufacturer", "field_t", "lang", "study_side", "sex", "age"]:
        print(c, out[c].astype(str).value_counts().head(12).to_dict())


if __name__ == "__main__":
    main()
