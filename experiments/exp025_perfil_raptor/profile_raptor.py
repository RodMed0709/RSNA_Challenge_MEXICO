#!/usr/bin/env python3
"""exp025_perfil_raptor -- per-stage profiler of the public Raptor/0.949 inference.

Third-party code under test: Kaggle notebook `nartaa/rsna-knee-0949-anatomical-mirror` (single
CoAtNet-2 arm `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` + anatomical L/R mirror TTA).
This script does NOT copy that code: it imports the notebook's own code cell (written to
--workdir) and its embedded `_fastread` module (base64 inside the cell), and re-times the same
steps by calling the notebook's functions in the same order. The deployed reader is
`study_lut`; the notebook's verbatim pydicom reader (`_make_reader` + `build_study`, called "v2"
inside _fastread) is profiled too as the naive reference. Each instrumented reader is checked to
be byte-identical to the notebook's own builder on the first studies.

Modes
  serial  per-study stage timing (one study at a time, page cache evicted before each study)
  pool    Kaggle-like prep throughput: notebook's fork Pool(nproc) of study_lut builders
  gpubench forward-pass variants on a few studies (chunk size, channels_last, TTA batching)

Usage (pod):  python -I profile_raptor.py --mode serial --n 50 --threads 112 --out res_112.json
              taskset -c 0-3 python -I profile_raptor.py --mode serial --n 50 --threads 4 --out res_4.json
"""
import argparse, base64, glob, importlib.util, io, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

DATA = "/workspace/data"
TSDIR = f"{DATA}/raw/train_series"
WEIGHTS = (f"{DATA}/public_assets/nartaa__rsna-knee-publication-swa-weights-20261007/"
           "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt")
L2_CSV = f"{DATA}/labels/public/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
REPO = "/workspace/users/rodrigo"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


# ---------------------------------------------------------------------------------------------
def load_notebook(nb_path, workdir):
    """Write the notebook's code cell + its embedded _fastread to workdir and import both."""
    nb = json.load(open(nb_path, encoding="utf-8"))
    src = max((''.join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"), key=len)
    os.makedirs(workdir, exist_ok=True)
    m = re.search(r'_FASTREAD_B64\s*=\s*"([A-Za-z0-9+/=]+)"', src)
    open(os.path.join(workdir, "_fastread.py"), "wb").write(base64.b64decode(m.group(1)))
    cell = os.path.join(workdir, "raptor_cell.py")
    open(cell, "w", encoding="utf-8").write(src)

    def _imp(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
        spec.loader.exec_module(mod); return mod
    C = _imp("raptor_cell", cell)
    FR = _imp("_fastread", os.path.join(workdir, "_fastread.py"))
    return C, FR


class Timer:
    def __init__(self):
        self.d = {}

    def add(self, k, v):
        self.d[k] = self.d.get(k, 0.0) + v


def evict(sid):
    """Drop the study's files from the page cache so every study is read cold (as on Kaggle)."""
    for f in glob.glob(f"{TSDIR}/{sid}/*/*.dcm"):
        try:
            fd = os.open(f, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        except OSError:
            pass


# ---- study_lut (deployed reader), instrumented copy of _fastread.build_study_batched ---------
def build_lut_timed(sid, SER, C, FR, ctx, T, meta):
    cfg = ctx.cfg; SLOTS, IMG = C.SLOTS, C.IMG; MAXS = sum(s[2] for s in SLOTS)
    t = time.perf_counter()
    rows = SER.get(sid, []); used = set(); plan = []
    for plane, fluid, k in SLOTS:
        r = FR._pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            plan.append((k, None)); continue
        used.add(r["SeriesInstanceUID"])
        plan.append((k, glob.glob(f"{TSDIR}/{sid}/{r['SeriesInstanceUID']}/*.dcm")))
    jobs = [(si, j, f) for si, (k, fs) in enumerate(plan) if fs for j, f in enumerate(fs)]
    keep = len(jobs) <= FR.keep_max()
    T.add("list", time.perf_counter() - t); t = time.perf_counter()
    outs = []; per, sels = {}, {}
    try:
        if jobs:
            with ThreadPoolExecutor(max_workers=min(cfg.hdr_threads, len(jobs))) as ex:
                outs = list(ex.map(lambda j: FR._hdr_prefix(ctx, j[2], keep), jobs))
        T.add("hdr_read", time.perf_counter() - t); t = time.perf_counter()
        for (si, j, f), o in zip(jobs, outs):
            per.setdefault(si, []).append(o)
        pj = []
        for si, (k, fs) in enumerate(plan):
            if not fs: continue
            order, med_ps = FR._records([o[0] for o in per[si]])
            sel = FR._picked(order, k)
            sels[si] = (sel, med_ps)
            pj += [(si, i) for i in dict.fromkeys(i for i, _ in sel)]
        T.add("select", time.perf_counter() - t); t = time.perf_counter()

        def _read(job):                                # I/O half of build_study_batched._one
            si, i = job
            rec, data, info, fd, full = per[si][i]
            if data is None: return None, None
            try:
                if not full:
                    data = (FR._read_rest(fd, data, len(data)) if fd is not None
                            else FR._read_whole(plan[si][1][i]))
                    info = FR._scan_full(data) if cfg.fast else None
            except Exception:
                return None, None
            return data, info
        raw = {}
        if pj:
            with ThreadPoolExecutor(max_workers=min(cfg.px_threads, len(pj))) as ex:
                raw = dict(zip(pj, ex.map(_read, pj)))
        T.add("px_read", time.perf_counter() - t); t = time.perf_counter()
        got = {j: (ctx.px(d, i if cfg.fast else None) if d is not None else (None, None))
               for j, (d, i) in raw.items()}           # decode half (walker / pydicom fallback)
        T.add("decode", time.perf_counter() - t)
    finally:
        for o in outs:
            if o[3] is not None:
                try: os.close(o[3])
                except OSError: pass
    meta.update(files=len(jobs), picks_unique=len(pj),
                bytes=int(sum(len(d) for d, _ in raw.values() if d is not None)),
                fallback=int(sum(1 for a, _ in got.values() if a is not None and not isinstance(a, FR.RawSlice))))
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0
    for si, (k, fs) in enumerate(plan):
        if not fs:
            idx += k; continue
        sel, med_ps = sels[si]
        res = FR._result(sel, {i: got[(si, i)] for i, _ in sel}, med_ps)
        idx = fill_lut_timed(vol, idx, MAXS, res, IMG, FR, T)
        if idx >= MAXS: break
    t = time.perf_counter()
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    T.add("crop_resize", time.perf_counter() - t)
    return vol, mask


def fill_lut_timed(vol, idx, MAXS, res, IMG, FR, T):
    """_fastread._fill_lut with timers: 'pct_norm' = exact 2/98 percentile from code counts +
    the per-series clip/normalise table; 'crop_resize' = table gather on the 140 mm crop +
    INTER_AREA resize to 384 + uint8."""
    import cv2
    t = time.perf_counter()
    arrs, pss, med_ps, _ = res
    valid = [a for a in arrs if a is not None]
    if (not valid or any(not isinstance(a, FR.RawSlice) for a in valid)
            or not FR._pct_selftest()):
        out = FR._fill_lut(vol, idx, MAXS, res, IMG)     # rare fallback: untimed split
        T.add("pct_norm", time.perf_counter() - t); return out
    groups = {}; n_total = 0; gkey = {}
    for a in valid:
        key = (a.dt, a.shift, a.rs, a.ri) + ((id(a),) if a.mono1 else ())
        gkey[id(a)] = key
        g = groups.get(key)
        if g is None: groups[key] = [a, a.cnt.copy()]
        else: g[1] += a.cnt
        n_total += a.size
    Vs, Cs, Tb = [], [], {}
    for key, (a0, cnt) in groups.items():
        full = a0.codes_to_float(FR._ALL_CODES)
        nz = np.flatnonzero(cnt); v = full[nz]
        if a0.mono1:
            vmax = v.max(); v = vmax - v; full = vmax - full
        Tb[key] = full; Vs.append(v); Cs.append(cnt[nz])
    V = np.concatenate(Vs); Cc = np.concatenate(Cs)
    if not np.isfinite(V).all():
        T.add("pct_norm", time.perf_counter() - t)
        return FR._fill_lut(vol, idx, MAXS, res, IMG)
    order = np.argsort(V, kind="stable"); V = V[order]; cum = np.cumsum(Cc[order])
    loq, hiq = FR._lerp_percentile(n_total, lambda r: V[int(np.searchsorted(cum, r, side="right"))])
    luts = {}
    for key in Tb:
        luts[key] = np.clip((Tb[key] - loq) / (hiq - loq + 1e-6), 0, 1)
    T.add("pct_norm", time.perf_counter() - t)
    t_norm = t_rs = 0.0
    for a, ps in zip(arrs, pss):
        if idx >= MAXS: break
        if a is None: idx += 1; continue
        t = time.perf_counter()
        lut = luts[gkey[id(a)]]
        p = ps if ps > 0 else med_ps
        h, w = a.shape; cpx = int(round(FR.CROP_MM / max(p, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        aw = lut.take(a.raw[y0:y0 + cpx, x0:x0 + cpx])  # normalisation = table gather on the crop
        t1 = time.perf_counter(); t_norm += t1 - t
        aw = cv2.resize(aw, (IMG, IMG), interpolation=cv2.INTER_AREA)
        vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        t_rs += time.perf_counter() - t1
    T.add("pct_norm", t_norm); T.add("crop_resize", t_rs)
    return idx


# ---- notebook's verbatim pydicom reader ("v2"), instrumented copy of C.build_study -----------
def build_v2_timed(sid, SER, C, T, meta):
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    order_and_meta, read_px, mm_crop_resize = C._make_reader()
    SLOTS, IMG, MAXS = C.SLOTS, C.IMG, C.MAXS

    def _hdr(f):                                         # == _make_reader._hdr
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, 'ImageOrientationPatient', None)
            ipp = getattr(h, 'ImagePositionPatient', None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                pos = float(np.dot(np.array(ipp, float), np.cross(r, c)))
            else:
                pos = float(getattr(h, 'InstanceNumber', 0) or 0)
            ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
            return (pos, f, ps, True)
        except Exception:
            return (0.0, f, 0.5, False)

    def _decode(b):                                      # == read_px, from bytes
        d = pydicom.dcmread(io.BytesIO(b))
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    rows = SER.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    nfiles = npicks = nbytes = 0
    for plane, fluid, k in SLOTS:
        r = C._pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        t = time.perf_counter()
        fs = glob.glob(f"{TSDIR}/{sid}/{r['SeriesInstanceUID']}/*.dcm")
        T.add("list", time.perf_counter() - t); t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=C.HDR_THREADS) as ex:
            out = list(ex.map(_hdr, fs))
        T.add("hdr_read", time.perf_counter() - t); t = time.perf_counter()
        recs = [(pos, f, ps) for pos, f, ps, _ in out]; ps_list = [ps for _, _, ps, ok in out if ok]
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        files = [(f, ps) for _, f, ps in recs]
        nfiles += len(files)
        if not files:
            idx += k; continue
        n = len(files); lo, hi = int(n * 0.02), int(n * 0.98) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        sel = [files[min(p, n - 1)] for p in picks]
        T.add("select", time.perf_counter() - t); t = time.perf_counter()
        # v2 re-reads every pick (also repeated ones); I/O and decode timed apart
        with ThreadPoolExecutor(max_workers=C.PX_THREADS) as ex:
            blobs = list(ex.map(lambda fp: open(fp[0], "rb").read(), sel))
        T.add("px_read", time.perf_counter() - t); t = time.perf_counter()
        npicks += len(sel); nbytes += sum(len(b) for b in blobs)
        arrs, pss = [], []
        for b, (fp, ps) in zip(blobs, sel):
            try: arrs.append(_decode(b)); pss.append(ps)
            except Exception: arrs.append(None); pss.append(med_ps)
        T.add("decode", time.perf_counter() - t); t = time.perf_counter()
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        normed = [None if a is None else np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1) for a in arrs]
        T.add("pct_norm", time.perf_counter() - t); t = time.perf_counter()
        for aw, ps in zip(normed, pss):
            if idx >= MAXS: break
            if aw is None: idx += 1; continue
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        T.add("crop_resize", time.perf_counter() - t)
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    meta.update(files=nfiles, picks_unique=npicks, bytes=int(nbytes), fallback=0)
    return vol, mask


# ---------------------------------------------------------------------------------------------
def pick_studies(n, seed=0):
    sys.path.insert(0, REPO)
    from src.knee.splits import load_split
    sp = load_split("split_v1_seed42")
    rng = np.random.default_rng(seed)
    gold = sorted(sp.loc[sp.fold == -1, "StudyInstanceUID"])
    f0 = sorted(sp.loc[sp.fold == 0, "StudyInstanceUID"])
    ng = min(len(gold), n // 2)
    g = list(rng.choice(gold, ng, replace=False)); o = list(rng.choice(f0, n - ng, replace=False))
    return g + o, set(g)


def load_ser():
    s = pd.read_csv(f"{DATA}/raw/train_series.csv")
    s["StudyInstanceUID"] = s["StudyInstanceUID"].astype(str)
    s["SeriesInstanceUID"] = s["SeriesInstanceUID"].astype(str)
    return {k: v.to_dict("records") for k, v in s.groupby("StudyInstanceUID")}


def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score
    aucs = {}
    for j, c in enumerate(LAB):
        m = ~np.isnan(y[:, j])
        yy = y[m, j]
        if m.sum() > 1 and 0 < yy.sum() < len(yy):
            aucs[c] = float(roc_auc_score(yy, p[m, j]))
    return (float(np.mean(list(aucs.values()))) if aucs else float("nan")), aucs


def sanity(sids, gold, probs):
    tr = pd.read_csv(f"{DATA}/raw/train.csv"); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    tr = tr.set_index("StudyInstanceUID")
    l2 = pd.read_csv(L2_CSV); l2["StudyInstanceUID"] = l2["StudyInstanceUID"].astype(str)
    l2 = l2.set_index("StudyInstanceUID")
    P = np.array([probs[s] for s in sids])
    gi = [i for i, s in enumerate(sids) if s in gold]
    yg = tr.loc[[sids[i] for i in gi], LAB].values.astype(float)
    ga, gper = macro_auc(yg, P[gi])
    have = [i for i, s in enumerate(sids) if s in l2.index]
    yl = (l2.loc[[sids[i] for i in have], LAB].values > 0.5).astype(float)
    la, lper = macro_auc(yl, P[have])
    return dict(gold_n=len(gi), gold_macro_auc=ga, gold_per_label=gper, gold_labels_used=len(gper),
                l2_n=len(have), l2_macro_auc=la, l2_per_label=lper)


# ---------------------------------------------------------------------------------------------
def run_serial(a, C, FR):
    import torch, cv2
    torch.set_num_threads(a.threads); cv2.setNumThreads(a.threads)
    dev = "cuda"
    SER = load_ser(); sids, gold = pick_studies(a.n)
    model, res = C.load_model(WEIGHTS, C.ARMS[0]["arch"], 384, dev)
    ctx = FR._Ctx(FR.Cfg(hdr_threads=C.FR_HDR, px_threads=C.FR_PX, lut=True))
    ref_lut = FR.make_builder("study_lut", C.IMG, C.SLOTS, hdr_threads=C.FR_HDR, px_threads=C.FR_PX)
    ref_v2 = C._make_reader()
    # warm-up (cudnn.benchmark autotune, CUDA context, imports) on a study outside the sample
    wu = [s for s in SER if s not in set(sids)][:1]
    for s in wu:
        vol, mask = ref_lut(s, SER, TSDIR)
        for _ in range(2):
            xw = C.gpu_windows(vol, mask, C.K_EVAL, res, dev); C.infer_probs(model, xw, dev)
    torch.cuda.synchronize()
    rows, probs, ident = [], {}, {"lut": [], "v2": []}
    for n_i, sid in enumerate(sids):
        for reader in a.readers:
            if a.evict: evict(sid)
            T = Timer(); meta = {}
            t0 = time.perf_counter()
            if reader == "lut":
                vol, mask = build_lut_timed(sid, SER, C, FR, ctx, T, meta)
            else:
                vol, mask = build_v2_timed(sid, SER, C, T, meta)
            t_prep = time.perf_counter() - t0
            if n_i < a.check:                            # byte-identity vs the notebook's builder
                rv = (ref_lut(sid, SER, TSDIR) if reader == "lut"
                      else C.build_study(sid, SER, TSDIR, ref_v2))
                ident[reader].append(bool(np.array_equal(rv[0], vol) and np.array_equal(rv[1], mask)))
            if reader == a.readers[0]:                   # GPU stages once per study
                torch.cuda.synchronize(); t = time.perf_counter()
                xw = C.gpu_windows(vol, mask, C.K_EVAL, res, dev)
                xm = C.gpu_windows(vol, mask, C.K_EVAL, res, dev, norm=C.NORM, mirror=True)
                torch.cuda.synchronize(); T.add("windows", time.perf_counter() - t); t = time.perf_counter()
                p = C.infer_probs(model, xw, dev)        # .cpu() inside -> synchronised
                T.add("fwd_plain", time.perf_counter() - t); t = time.perf_counter()
                q = C.infer_probs(model, xm, dev)
                T.add("fwd_mirror", time.perf_counter() - t)
                probs[sid] = (0.5 * (p + q)).tolist()
            rows.append(dict(sid=sid, gold=sid in gold, reader=reader, prep=t_prep,
                             total=time.perf_counter() - t0, **T.d, **meta))
        if (n_i + 1) % 10 == 0:
            print(f"[serial] {n_i+1}/{len(sids)}", flush=True)
    df = pd.DataFrame(rows)
    out = dict(mode="serial", threads=a.threads, n=len(sids), readers=a.readers, evict=a.evict,
               identical=ident, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               sanity=sanity(sids, gold, probs),
               mean=df.groupby("reader").mean(numeric_only=True).to_dict("index"),
               median=df.groupby("reader").median(numeric_only=True).to_dict("index"),
               rows=rows)
    return out


def run_pool(a, C, FR):
    """Kaggle-like prep: the notebook's own fork Pool(nproc) of study_lut builders, timed wall."""
    import multiprocessing as mp
    SER = load_ser(); sids, _ = pick_studies(a.n)
    if a.evict:
        for s in sids: evict(s)
    C._SER, C._TSDIR = SER, TSDIR
    nproc = a.procs or max(4, min(96, (os.cpu_count() or 8) - 8))
    t = time.perf_counter()
    with mp.get_context("fork").Pool(nproc, initializer=C._pre_init) as pool:
        n = sum(1 for _ in pool.imap_unordered(C._pre_one, sids, chunksize=2))
    wall = time.perf_counter() - t
    return dict(mode="pool", procs=nproc, n=n, affinity=len(os.sched_getaffinity(0)),
                wall=wall, s_per_study=wall / n)


def run_gpubench(a, C, FR):
    """Forward variants on a few studies: chunk size, channels_last, plain+mirror in one batch."""
    import torch
    dev = "cuda"
    SER = load_ser(); sids, _ = pick_studies(a.n)
    sids = sids[: a.bench_n]
    model, res = C.load_model(WEIGHTS, C.ARMS[0]["arch"], 384, dev)
    ref = FR.make_builder("study_lut", C.IMG, C.SLOTS, hdr_threads=C.FR_HDR, px_threads=C.FR_PX)
    vols = [ref(s, SER, TSDIR) for s in sids]
    xs = [(C.gpu_windows(v, m, C.K_EVAL, res, dev),
           C.gpu_windows(v, m, C.K_EVAL, res, dev, norm=C.NORM, mirror=True)) for v, m in vols]
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16), FlopCounterMode(display=False) as fc:
        model(xs[0][0].unsqueeze(0))
    gflop_pass = fc.get_total_flops() / 1e9
    orig_encode = C.RaptorClassifier.encode

    def make_encode(chunk, cl=False):
        def encode(self, x):
            B, K = x.shape[:2]; flat = x.flatten(0, 1)
            if cl: flat = flat.contiguous(memory_format=torch.channels_last)
            f = torch.cat([self.backbone(c) for c in flat.split(chunk)], dim=0)
            return f.view(B, K, -1)
        return encode

    @torch.no_grad()
    def fwd(x):
        with torch.autocast("cuda", dtype=torch.float16):
            return torch.sigmoid(model(x).float())

    def bench(name, fn, reps=3):
        for _ in range(2): fn(xs[0])                     # warm-up / autotune per shape
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(reps):
            for x in xs: fn(x)
        torch.cuda.synchronize()
        return name, (time.perf_counter() - t) / (reps * len(xs))
    res_ = {}
    ref_p = None
    for chunk in (16, 32, 47, 94):
        C.RaptorClassifier.encode = make_encode(chunk)
        k, v = bench(f"tta2_chunk{chunk}", lambda x: (fwd(x[0].unsqueeze(0)), fwd(x[1].unsqueeze(0))))
        res_[k] = v
    # plain+mirror stacked as a batch of 2 studies (188 windows, chunk 16 / 47)
    for chunk in (16, 47):
        C.RaptorClassifier.encode = make_encode(chunk)
        k, v = bench(f"tta_batched_chunk{chunk}", lambda x: fwd(torch.stack([x[0], x[1]])))
        res_[k] = v
    C.RaptorClassifier.encode = make_encode(16)
    k, v = bench("notta_chunk16", lambda x: fwd(x[0].unsqueeze(0))); res_[k] = v
    model.to(memory_format=torch.channels_last)
    for chunk in (16, 47):
        C.RaptorClassifier.encode = make_encode(chunk, cl=True)
        k, v = bench(f"tta2_chunk{chunk}_channels_last",
                     lambda x: (fwd(x[0].unsqueeze(0)), fwd(x[1].unsqueeze(0)))); res_[k] = v
    C.RaptorClassifier.encode = orig_encode
    return dict(mode="gpubench", n=len(sids), gpu=torch.cuda.get_device_name(0),
                gflop_per_pass_94win=gflop_pass, s_per_study=res_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["serial", "pool", "gpubench"], default="serial")
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--readers", default="lut,v2")
    ap.add_argument("--check", type=int, default=3)
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--bench-n", type=int, default=5)
    ap.add_argument("--no-evict", dest="evict", action="store_false")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); a.readers = a.readers.split(",")
    C, FR = load_notebook(a.notebook, a.workdir)
    t = time.time()
    out = {"serial": run_serial, "pool": run_pool, "gpubench": run_gpubench}[a.mode](a, C, FR)
    out["affinity_cpus"] = len(os.sched_getaffinity(0)); out["elapsed"] = time.time() - t
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=float))


if __name__ == "__main__":
    main()
