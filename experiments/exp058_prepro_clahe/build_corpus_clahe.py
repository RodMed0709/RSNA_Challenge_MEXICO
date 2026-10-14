"""exp058 (copied from exp049; adds build_meta["prepro"]) - corpus CLAHE: aplica CLAHE de OpenCV (clip 2.0, tiles 8x8) a cada corte valido del corpus Raptor
d96/384 (uint8) y lo guarda en OUT (disco local del pod). Script auxiliar del experimento (no entrena).

Entrada: SRC/vols.npy (N, 96, 384, 384) uint8, SRC/masks.npy, SRC/ids.npy, SRC/build_meta.json (build_corpus96.py).
Salida:  OUT/vols.npy (misma forma, CLAHE por corte; cortes no validos quedan a cero), masks/ids copiados,
         build_meta.json = el de SRC + {"clahe": {...}, "clahe_seconds", "clahe_ms_per_study"}.
El kernel de inferencia aplica lo mismo: cv2.createCLAHE(2.0, (8, 8)).apply(corte_uint8_384) por corte valido,
antes del recorte central / ventanas.

Uso:  python build_corpus_clahe.py --src /root/corpus96_384 --out /root/corpus_clahe --procs 16 [--limit 40]
"""
import argparse, json, os, shutil, time
from multiprocessing import Pool

import numpy as np

CLIP = 2.0
TILES = (8, 8)
_G = {}


def _init(src, out):
    import cv2
    cv2.setNumThreads(1)
    _G["cl"] = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=TILES)
    _G["src"] = np.load(os.path.join(src, "vols.npy"), mmap_mode="r")
    _G["dst"] = np.load(os.path.join(out, "vols.npy"), mmap_mode="r+")
    _G["masks"] = np.load(os.path.join(src, "masks.npy"))


def _one(row):
    vol = np.array(_G["src"][row])
    t = time.time()
    out = np.zeros_like(vol)
    for d in np.where(_G["masks"][row] > 0)[0]:
        out[d] = _G["cl"].apply(vol[d])
    dt = time.time() - t
    _G["dst"][row] = out
    return row, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/corpus96_384")
    ap.add_argument("--out", default="/root/corpus_clahe")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = np.load(os.path.join(a.src, "vols.npy"), mmap_mode="r")
    N = a.limit or src.shape[0]
    dst = np.lib.format.open_memmap(os.path.join(a.out, "vols.npy"), mode="w+", dtype=np.uint8,
                                    shape=(src.shape[0],) + src.shape[1:])
    del dst, src
    t0 = time.time(); done = 0; cl_s = []
    with Pool(a.procs, initializer=_init, initargs=(a.src, a.out)) as pool:
        for row, dt in pool.imap_unordered(_one, range(N), chunksize=4):
            cl_s.append(dt); done += 1
            if done % 200 == 0 or done == N:
                el = time.time() - t0
                print(f"{done}/{N} | {el:.0f}s | {done/el:.2f} est/s | ETA {(N-done)/(done/el)/60:.1f} min "
                      f"| CLAHE {1000*np.mean(cl_s):.1f} ms/estudio (1 hilo)", flush=True)
    for f in ("masks.npy", "ids.npy"):
        shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    meta = json.load(open(os.path.join(a.src, "build_meta.json")))
    meta.update({"clahe": {"clip": CLIP, "tiles": list(TILES), "src": a.src, "lib": "cv2.createCLAHE"},
                 "clahe_seconds": time.time() - t0, "clahe_procs": a.procs, "clahe_n": N,
                 "clahe_ms_per_study": 1000 * float(np.mean(cl_s))})
    meta["prepro"] = {"method": "clahe", "ref": "Zuiderveld 1994", "clip": CLIP, "tiles": list(TILES), "src": a.src, "n": N,
                      "procs": a.procs, "build_seconds": meta["clahe_seconds"], "ms_per_study_1thread": meta["clahe_ms_per_study"],
                      "ms_per_study_p95": 1000 * float(np.percentile(cl_s, 95))}
    json.dump(meta, open(os.path.join(a.out, "build_meta.json"), "w"), indent=1)
    print("DONE", json.dumps({k: v for k, v in meta.items() if k != "errors"}), flush=True)


if __name__ == "__main__":
    main()
