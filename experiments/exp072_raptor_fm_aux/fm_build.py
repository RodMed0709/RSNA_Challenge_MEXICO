"""Download the fastMRI knee single-coil volumes listed by fastMRI+ (knee_file_list.csv) one by one from Kaggle copies,
extract reconstruction_rss (fallback reconstruction_esc), flip up-down (fastMRI+ label convention, ExampleScripts/example.ipynb),
normalise per volume to uint8 with p2/p98 (same convention as our corpus), resize 320 -> 384, save vol/<file>.npy, delete the h5.
usage: python fm_build.py MANIFEST FILE_LIST OUT_DIR TMP_DIR [workers]"""
import json, os, sys, time, zipfile, threading, queue, traceback
import numpy as np, h5py, cv2
from kaggle.api.kaggle_api_extended import KaggleApi

man = json.load(open(sys.argv[1]))
files = [l.strip() for l in open(sys.argv[2]) if l.strip().startswith("file")]
out, tmp = sys.argv[3], sys.argv[4]
W = int(sys.argv[5]) if len(sys.argv) > 5 else 6
os.makedirs(out + "/vol", exist_ok=True); os.makedirs(tmp, exist_ok=True)
api = KaggleApi(); api.authenticate()
lock = threading.Lock()
q = queue.Queue()
todo = [f for f in files if f + ".h5" in man and not os.path.exists(f"{out}/vol/{f}.npy")]
for f in todo: q.put(f)
print("todo", len(todo), "of", len(files), flush=True)
t0 = time.time(); done = [0]


def one(f, wd):
    ds = man[f + ".h5"][0]
    for att in range(4):
        try:
            api.dataset_download_file(ds, f + ".h5", path=wd, quiet=True)
            break
        except Exception as e:
            if att == 3: raise
            time.sleep(10 * (att + 1))
    p = os.path.join(wd, f + ".h5")
    if not os.path.exists(p) and os.path.exists(p + ".zip"):
        with zipfile.ZipFile(p + ".zip") as z: z.extractall(wd)
        os.remove(p + ".zip")
    with h5py.File(p, "r") as h:
        key = "reconstruction_rss" if "reconstruction_rss" in h else "reconstruction_esc"
        v = h[key][()].astype(np.float32)
        acq = str(h.attrs.get("acquisition", ""))
    os.remove(p)
    v = v[:, ::-1, :]
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    u = np.clip((v - lo) / max(hi - lo, 1e-8), 0, 1) * 255
    u = np.stack([cv2.resize(s, (384, 384), interpolation=cv2.INTER_LINEAR) for s in u]).round().astype(np.uint8)
    np.save(f"{out}/vol/{f}.npy.tmp.npy", u); os.replace(f"{out}/vol/{f}.npy.tmp.npy", f"{out}/vol/{f}.npy")
    return {"file": f, "dataset": ds, "key": key, "acq": acq, "n_slices": int(u.shape[0]), "src_hw": list(v.shape[1:]),
            "p2": float(lo), "p98": float(hi)}


def worker(i):
    wd = os.path.join(tmp, "w%d" % i); os.makedirs(wd, exist_ok=True)
    while True:
        try: f = q.get_nowait()
        except queue.Empty: return
        try:
            m = one(f, wd)
            with lock:
                open(out + "/meta.jsonl", "a").write(json.dumps(m) + "\n"); done[0] += 1
                if done[0] % 20 == 0:
                    print(f"{done[0]}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
        except Exception:
            with lock: print("ERR", f, traceback.format_exc()[-400:], flush=True)
            for x in os.listdir(wd): os.remove(os.path.join(wd, x))


th = [threading.Thread(target=worker, args=(i,)) for i in range(W)]
[t.start() for t in th]; [t.join() for t in th]
print("DONE", done[0], f"{time.time()-t0:.0f}s", flush=True)
