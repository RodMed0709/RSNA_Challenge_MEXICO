"""exp084 prediction-preserving speedups on top of exp080-ctrl224-fast (scored body = 2nd code cell).
X1 early prep: a new FIRST code cell forks the 6 prep processes and starts streaming studies before torch/timm are
   imported, the checkpoint is hashed and the models are loaded (main() consumes that iterator).
X2 workers return only the 320x320 centre crop the model uses (uint8, same values) -> ~30 % less IPC per study.
X3 cv2.setNumThreads(1) in the prep workers (6 procs on 4 vCPU; same output).
X4 GPU windows from the cropped volume; float conversion by division instead of a 256-entry LUT + int64 index
   (IEEE float32 division, same values as the CPU-built LUT).
X5 (variant 'b') torch.backends.cudnn.benchmark = True (fixed (64,3,224,224) shape; may change fp16 rounding).
apply(body, root_expr, cudnn_bench) -> (early_cell_source, new_body)."""


def rep(t, a, b, n=1):
    assert t.count(a) == n, (a[:70], t.count(a))
    return t.replace(a, b)


def early_cell(fastread_line, root_expr):
    return '''# [exp084] early prep: fork the DICOM prep processes and start streaming before torch/timm import and model load.
import os, sys, glob, time, json, gc, threading, queue
import numpy as np, pandas as pd
import multiprocessing as mp
T_EARLY0 = time.time()
IMG = 384
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(s[2] for s in SLOTS)
FASTREAD_VARIANT = 'study_lut'
FR_HDR, FR_PX = 32, 16
CROP_FRAC = 320 / 384
EARLY_NPROC = 6
''' + fastread_line + '''
def _load_fastread():
    import base64 as _b, importlib.util as _iu
    p = '/kaggle/working/_fastread.py' if os.path.isdir('/kaggle/working') else os.path.join(os.getcwd(), '_fastread.py')
    if not os.path.exists(p):
        open(p, 'wb').write(_b.b64decode(_FASTREAD_B64))
    spec = _iu.spec_from_file_location('_fastread', p); m = _iu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _early_root():
    for b in ["/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"]:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")


_ESER = None; _ETSDIR = None; _EREADER = None
def _early_init():
    global _EREADER
    try:
        import cv2; cv2.setNumThreads(1)   # [exp084] X3: one cv2 thread per prep process (same output)
    except Exception:
        pass
    _EREADER = _load_fastread().make_builder(FASTREAD_VARIANT, IMG, SLOTS[:3], hdr_threads=FR_HDR, px_threads=FR_PX)


def _early_one(sid):
    _t0 = time.time(); _c0 = time.process_time()
    try:
        vol, mask = _EREADER(sid, _ESER, _ETSDIR)
    except Exception as e:
        print(f"  [pre] {sid[:16]} FAILED ({type(e).__name__}: {e})", flush=True)
        vol = np.zeros((MAXS, IMG, IMG), np.uint8); mask = np.zeros(MAXS, np.uint8)
    H = vol.shape[-1]; ch = int(round(H * CROP_FRAC)); y0 = (H - ch) // 2
    vol = np.ascontiguousarray(vol[:, y0:y0 + ch, y0:y0 + ch])   # [exp084] X2: only the centre crop the model reads
    return sid, vol, mask, (time.time() - _t0, time.process_time() - _c0)


EARLY_ROOT = ''' + root_expr + '''
EARLY_TSDIR = EARLY_ROOT + "/test_series"
if not os.path.isdir(EARLY_TSDIR):
    EARLY_TSDIR = EARLY_ROOT + "/test_images"
_et = pd.read_csv(EARLY_ROOT + "/test.csv"); _et["StudyInstanceUID"] = _et["StudyInstanceUID"].astype(str)
EARLY_IDS = _et["StudyInstanceUID"].tolist()
_es = pd.read_csv(EARLY_ROOT + "/test_series.csv")
_es["StudyInstanceUID"] = _es["StudyInstanceUID"].astype(str); _es["SeriesInstanceUID"] = _es["SeriesInstanceUID"].astype(str)
_ESER = {k: v.to_dict("records") for k, v in _es.groupby("StudyInstanceUID")}
_ETSDIR = EARLY_TSDIR
EARLY_POOL = mp.get_context("fork").Pool(EARLY_NPROC, initializer=_early_init)
EARLY_IT = EARLY_POOL.imap_unordered(_early_one, EARLY_IDS, chunksize=2)
print(f"[early] {len(EARLY_IDS)} studies streaming from {EARLY_NPROC} prep processes | {time.time() - T_EARLY0:.1f}s", flush=True)
'''


GPU_WIN_X = '''
def gpu_windows(vol, mask, k, res, dev, norm=NORM):
    """[exp084] X4: vol is the 320x320 centre crop (prep workers crop); float conversion by IEEE division (same values
    as the CPU float32 LUT of the original), no int64 index tensor."""
    D = mask.shape[0]
    cs = np.array([max(1, min(c, D - 2)) for c in _eval_centers(mask, D, k)])
    if vol.shape[-1] == IMG or not (norm == 'imagenet' and CROP_FRAC < 1.0):
        return _gpu_windows_orig(vol, mask, k, res, dev, norm)
    vg = torch.from_numpy(vol).to(dev)
    if int(cs.max()) + 1 >= vg.shape[0]:
        vg = torch.cat([vg, vg.new_zeros((int(cs.max()) + 2 - vg.shape[0],) + tuple(vg.shape[1:]))], 0)
    tri = vg[torch.from_numpy(np.stack([cs - 1, cs, cs + 1], 1)).to(dev)]          # (k,3,320,320) uint8
    x = tri.to(torch.float32).div_(255.0)
    x = F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)
    return (x - _MEAN.to(dev)) / _STD.to(dev)

'''


def apply(body, root_expr, cudnn_bench, bench=False):
    lines = body.split("\n")
    fr = [l for l in lines if l.startswith("_FASTREAD_B64 = ")]
    assert len(fr) == 1
    body = body.replace(fr[0] + "\n", "", 1)
    # X4: the fast gpu_windows of exp080-ctrl224-fast becomes the crop-aware one
    a = body.index("\ndef gpu_windows(vol, mask, k, res, dev, norm=NORM):\n")
    b = body.index("\n@torch.no_grad()\ndef infer_probs(")
    body = body[:a] + "\n" + GPU_WIN_X + body[b:]
    if cudnn_bench:
        body = rep(body, "torch.backends.cudnn.benchmark = False   # exp027: no per-thread cudnn autotune",
                   "torch.backends.cudnn.benchmark = True   # [exp084] X5: fixed (64,3,224,224) shape, autotune once per GPU")
    # X1: main consumes the early iterator instead of creating the pool
    old_loop = ("    with mp.get_context(\"fork\").Pool(nproc, initializer=_pre_init) as pool:\n"
                "        for i, (sid, vol, mask) in enumerate(pool.imap_unordered(_pre_one, test_ids, chunksize=2)):\n"
                "            q.put((sid, vol, mask))\n")
    new_loop = ("    assert EARLY_IDS == test_ids, \"[exp084] early prep ids differ from test.csv\"\n"
                "    nproc = EARLY_NPROC\n"
                "    if True:   # [exp084] X1: prep processes started in the first cell\n"
                "        for i, (sid, vol, mask, _pt) in enumerate(EARLY_IT):\n"
                + ("            PRE_T[sid] = _pt; QS.append(q.qsize())\n" if bench else "")
                + "            q.put((sid, vol, mask))\n")
    body = rep(body, old_loop, new_loop)
    body = rep(body, "    for _ in devs: q.put(None)\n", "    EARLY_POOL.close()\n    for _ in devs: q.put(None)\n")
    return early_cell(fr[0], root_expr), body
