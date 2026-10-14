#!/usr/bin/env python3
"""exp026_t4_variantes -- Raptor/0.949 inference timed on Kaggle's real T4 x2 + disk, 5 forward variants.

Runs INSIDE a Kaggle kernel. The previous kernel cells write (by make_kernel.py):
  /kaggle/working/raptor_cell.py   the code cell of `nartaa/rsna-knee-0949-anatomical-mirror` (verbatim)
  /kaggle/working/exp026_sids.json {"main": [...150], "serial": [...30], "warmup": [...2]}
Nothing of the third-party code is copied here: it is imported and its functions are called in the
notebook's order. Stage instrumentation of the study_lut reader is the one from exp025.

Variants (same studies, same weights, fp16 autocast):
  V0  notebook as is: 94 windows, anatomical mirror TTA (2 passes), chunks of 16
  V1  V0 + channels_last (model + every window chunk)
  V2  V1 without TTA
  V3  V1 with mirror TTA on the sagittal windows only (feature-level bag: mirrored sagittal features +
      plain coronal/axial features; mean of the two sigmoids)
  V4  V1 with 47 windows (K_EVAL=47, every 2nd) + TTA
  V0b V0 repeated at the end (drift / noise control)

Outputs in /kaggle/working: exp026_results.json, preds_<V>.csv, gpu_util.csv
"""
import glob, importlib.util, json, os, queue, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd

W = "/kaggle/working"
T_START = time.time()


def log(*a):
    print(f"[{time.time() - T_START:7.1f}s]", *a, flush=True)


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


t = time.time()
C = _imp("raptor_cell", f"{W}/raptor_cell.py")            # imports torch/timm, sets cudnn.benchmark
FR = C._load_fastread()                                  # writes /kaggle/working/_fastread.py
import torch
IMPORT_S = time.time() - t
SIDS = json.load(open(f"{W}/exp026_sids.json"))
_N = int(os.environ.get("EXP026_N", "0"))            # smoke only: cut the study lists
if _N:
    SIDS = {k: v[:_N] for k, v in SIDS.items()}
LAB = C.LAB
RES = C.INPUT_RES


# ---------------------------------------------------------------------------------------------
class Timer:
    def __init__(self): self.d = {}
    def add(self, k, v): self.d[k] = self.d.get(k, 0.0) + v


def build_lut_timed(sid, SER, TSDIR, ctx, T, meta):
    """exp025 instrumented copy of _fastread.build_study_batched (study_lut)."""
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

        def _read(job):
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
               for j, (d, i) in raw.items()}
        T.add("decode", time.perf_counter() - t)
    finally:
        for o in outs:
            if o[3] is not None:
                try: os.close(o[3])
                except OSError: pass
    meta.update(files=len(jobs), picks_unique=len(pj),
                bytes=int(sum(len(d) for d, _ in raw.values() if d is not None)))
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0
    for si, (k, fs) in enumerate(plan):
        if not fs:
            idx += k; continue
        sel, med_ps = sels[si]
        res = FR._result(sel, {i: got[(si, i)] for i, _ in sel}, med_ps)
        idx = fill_lut_timed(vol, idx, MAXS, res, IMG, T)
        if idx >= MAXS: break
    t = time.perf_counter()
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    T.add("crop_resize", time.perf_counter() - t)
    return vol, mask


def fill_lut_timed(vol, idx, MAXS, res, IMG, T):
    """exp025 instrumented copy of _fastread._fill_lut ('pct_norm' / 'crop_resize')."""
    import cv2
    t = time.perf_counter()
    arrs, pss, med_ps, _ = res
    valid = [a for a in arrs if a is not None]
    if (not valid or any(not isinstance(a, FR.RawSlice) for a in valid)
            or not FR._pct_selftest()):
        out = FR._fill_lut(vol, idx, MAXS, res, IMG)
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
    luts = {key: np.clip((Tb[key] - loq) / (hiq - loq + 1e-6), 0, 1) for key in Tb}
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
        aw = lut.take(a.raw[y0:y0 + cpx, x0:x0 + cpx])
        t1 = time.perf_counter(); t_norm += t1 - t
        aw = cv2.resize(aw, (IMG, IMG), interpolation=cv2.INTER_AREA)
        vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        t_rs += time.perf_counter() - t1
    T.add("pct_norm", t_norm); T.add("crop_resize", t_rs)
    return idx


# ---- forward variants: fn(model, vol, mask, dev) -> (probs[12], t_windows, t_forward) ----------
def _sync(d): torch.cuda.synchronize(d)


def v_notebook(model, vol, mask, d):                      # V0: exactly the notebook's consumer body
    t = time.perf_counter()
    xw = C.gpu_windows(vol, mask, C.K_EVAL, RES, d)
    xm = C.gpu_windows(vol, mask, C.K_EVAL, RES, d, norm=C.NORM, mirror=True)
    _sync(d); t1 = time.perf_counter()
    p = C.infer_pair_probs(model, xw, xm, d)
    return p, t1 - t, time.perf_counter() - t1


def make_tta(k):                                          # V1 (k=94) / V4 (k=47)
    def fn(model, vol, mask, d):
        t = time.perf_counter()
        xw = C.gpu_windows(vol, mask, k, RES, d)
        xm = C.gpu_windows(vol, mask, k, RES, d, norm=C.NORM, mirror=True)
        _sync(d); t1 = time.perf_counter()
        p = 0.5 * (C.infer_probs(model, xw, d) + C.infer_probs(model, xm, d))
        return p, t1 - t, time.perf_counter() - t1
    return fn


def v_notta(model, vol, mask, d):                         # V2
    t = time.perf_counter()
    xw = C.gpu_windows(vol, mask, C.K_EVAL, RES, d)
    _sync(d); t1 = time.perf_counter()
    p = C.infer_probs(model, xw, d)
    return p, t1 - t, time.perf_counter() - t1


def v_sagtta(model, vol, mask, d):                        # V3
    t = time.perf_counter()
    D = vol.shape[0]
    cs = np.array([max(1, min(c, D - 2)) for c in C._eval_centers(mask, D, C.K_EVAL)])
    s_, _ = C._slot_index(D)
    sagi = torch.from_numpy(np.flatnonzero(np.isin(s_[cs], C.SAGITTAL_SLOTS))).to(d)
    xw = C.gpu_windows(vol, mask, C.K_EVAL, RES, d)
    xm = C.gpu_windows(vol, mask, C.K_EVAL, RES, d, norm=C.NORM, mirror=True)[sagi]
    _sync(d); t1 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        f = model.encode(xw.unsqueeze(0).to(d))
        fm = f.clone()
        if len(sagi):
            fm[0, sagi] = model.encode(xm.unsqueeze(0))[0].to(fm.dtype)
        lp = model.head(f).float(); lq = model.head(fm).float()
        assert torch.isfinite(lp).all() and torch.isfinite(lq).all()
        p = 0.5 * (torch.sigmoid(lp) + torch.sigmoid(lq))
    return p[0].cpu().numpy(), t1 - t, time.perf_counter() - t1


ORIG_ENCODE = C.RaptorClassifier.encode


def encode_cl(self, x):                                   # notebook encode + channels_last chunks
    B, K = x.shape[:2]
    flat = x.flatten(0, 1)
    f = torch.cat([self.backbone(c.contiguous(memory_format=torch.channels_last))
                   for c in flat.split(16)], dim=0)
    return f.view(B, K, -1)


def set_layout(models, cl):
    C.RaptorClassifier.encode = encode_cl if cl else ORIG_ENCODE
    fmt = torch.channels_last if cl else torch.contiguous_format
    for d, m in models.items():
        m.to(memory_format=fmt)


# ---- nvidia-smi sampler -----------------------------------------------------------------------
class GpuMon:
    def __init__(self):
        self.rows = []; self.phase = "init"
        try:
            self.p = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,clocks.sm,temperature.gpu",
                 "--format=csv,noheader,nounits", "-lms", "500"],
                stdout=subprocess.PIPE, text=True)
            threading.Thread(target=self._read, daemon=True).start()
        except Exception as e:
            log("nvidia-smi unavailable", e); self.p = None

    def _read(self):
        for line in self.p.stdout:
            try:
                i, u, m, clk, tmp = [x.strip() for x in line.split(",")]
                self.rows.append((time.time() - T_START, self.phase, int(i), float(u), float(m),
                                  float(clk), float(tmp)))
            except Exception:
                pass

    def summary(self, phase):
        df = pd.DataFrame(self.rows, columns=["t", "phase", "gpu", "util", "mem", "clk", "temp"])
        df = df[df.phase == phase]
        return {int(g): dict(util_mean=round(float(x.util.mean()), 1), mem_max=float(x.mem.max()),
                             clk_mean=round(float(x.clk.mean())), temp_max=float(x.temp.max()),
                             n=int(len(x))) for g, x in df.groupby("gpu")}

    def stop(self):
        if self.p is not None: self.p.terminate()
        pd.DataFrame(self.rows, columns=["t", "phase", "gpu", "util", "mem", "clk", "temp"]).to_csv(
            f"{W}/gpu_util.csv", index=False)


# ---- pipeline runner (the notebook's main: fork Pool prep -> queue -> one thread per GPU) ------
def run_variant(name, fn, sids, models, mon, source, stash=None):
    """source='disk': notebook pipeline with Pool prep; source='ram': feed stashed volumes."""
    devs = list(models)
    q = queue.Queue(maxsize=C.QUEUE_MAX); probs = {}; lock = threading.Lock()
    st = {d: dict(n=0, busy=0.0, windows=0.0, fwd=0.0, wait=0.0, fail=0) for d in devs}
    per = {d: [] for d in devs}                           # per-study busy (1st = cudnn autotune of a new thread)

    def consumer(d):
        while True:
            tw = time.perf_counter(); item = q.get(); st[d]["wait"] += time.perf_counter() - tw
            if item is None: q.task_done(); return
            sid, vol, mask = item
            t = time.perf_counter()
            try:
                p, a, b = fn(models[d], vol, mask, d)
                st[d]["windows"] += a; st[d]["fwd"] += b
            except Exception as e:
                log(f"  [{name} {d}] {sid[:16]} FAIL {type(e).__name__}: {e}")
                p = np.full(len(LAB), 0.5, np.float32); st[d]["fail"] += 1
            bt = time.perf_counter() - t; per[d].append(bt)
            st[d]["busy"] += bt; st[d]["n"] += 1
            with lock:
                probs[sid] = np.asarray(p, np.float64)
                if stash is not None and source == "disk": stash[sid] = (vol, mask)
            q.task_done()

    mon.phase = f"{name}_{source}"
    ths = [threading.Thread(target=consumer, args=(d,), daemon=True) for d in devs]
    t0 = time.perf_counter()
    for th in ths: th.start()
    if source == "disk":
        C._SER, C._TSDIR = SER, TSDIR
        with mp.get_context("fork").Pool(NPROC, initializer=C._pre_init) as pool:
            for sid, vol, mask in pool.imap_unordered(C._pre_one, sids, chunksize=2):
                q.put((sid, vol, mask))
    else:
        for sid in sids:
            q.put((sid,) + stash[sid])
    t_prep = time.perf_counter() - t0
    for _ in devs: q.put(None)
    q.join()
    for th in ths: th.join()
    wall = time.perf_counter() - t0
    n = len(sids)
    out = dict(variant=name, source=source, n=n, wall=wall, s_per_study=wall / n, t_feed=t_prep,
               per_gpu={d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
                        for d, s in st.items()},
               nvsmi=mon.summary(mon.phase))
    for d, s in st.items():
        out["per_gpu"][d]["gpu_s_per_study"] = round(s["busy"] / max(1, s["n"]), 4)
        out["per_gpu"][d]["busy_frac"] = round(s["busy"] / wall, 3)
        if per[d]:
            med = float(np.median(per[d][1:])) if len(per[d]) > 1 else per[d][0]
            out["per_gpu"][d].update(first_study_s=round(per[d][0], 3), steady_gpu_s_per_study=round(med, 4))
    # cudnn.benchmark autotunes again in every new consumer thread (per-thread cache): the notebook pays it
    # once per GPU per run. Steady wall = wall minus the largest per-GPU excess of the first study.
    exc = max([out["per_gpu"][d].get("first_study_s", 0) - out["per_gpu"][d].get("steady_gpu_s_per_study", 0)
               for d in devs] + [0.0])
    out["autotune_excess_s"] = round(exc, 2)
    out["steady_s_per_study"] = (wall - exc) / n
    log(f"{name:4s} {source:4s} wall {wall:7.1f}s  {wall / n:.3f} s/study  steady {out['steady_s_per_study']:.3f}  feed {t_prep:6.1f}s  " +
        "  ".join(f"{d}: n={s['n']} gpu={s['busy'] / max(1, s['n']):.3f}s/st wait={s['wait']:.0f}s"
                  for d, s in st.items()))
    pd.DataFrame([[s] + list(probs[s]) for s in sids], columns=["StudyInstanceUID"] + LAB).to_csv(
        f"{W}/preds_{name}_{source}.csv", index=False)
    return out


def warmup(fn, models, vol, mask, reps=2):
    t = time.perf_counter()
    for d, m in models.items():
        for _ in range(reps): fn(m, vol, mask, d)
        _sync(d)
    return time.perf_counter() - t


# ---------------------------------------------------------------------------------------------
def main():
    global SER, TSDIR, NPROC
    R = dict(exp="exp026_t4_variantes", import_s=IMPORT_S)
    ngpu = torch.cuda.device_count()
    R["env"] = dict(torch=torch.__version__, timm=__import__("timm").__version__, cpu_count=os.cpu_count(),
                    affinity=len(os.sched_getaffinity(0)), ngpu=ngpu,
                    gpus=[torch.cuda.get_device_name(i) for i in range(ngpu)],
                    ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1))
    log("env", R["env"])
    ROOT = C.find_test_root()
    TSDIR = next(p for p in (ROOT + "/train_series", ROOT + "/train_images") if os.path.isdir(p))
    s = pd.read_csv(ROOT + "/train_series.csv")
    s["StudyInstanceUID"] = s["StudyInstanceUID"].astype(str)
    s["SeriesInstanceUID"] = s["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in s.groupby("StudyInstanceUID")}
    R["root"], R["tsdir"] = ROOT, TSDIR
    try:
        R["mounts"] = [l for l in open("/proc/mounts").read().splitlines() if "/kaggle/input" in l][:10]
    except Exception:
        pass
    NPROC = max(4, min(96, (os.cpu_count() or 8) - 8))          # notebook's formula
    R["nproc"] = NPROC
    for k in ("main", "serial", "warmup"):
        miss = [x for x in SIDS[k] if x not in SER]
        assert not miss, f"{k}: {len(miss)} studies missing from train_series.csv"
    log(f"root {ROOT} | tsdir {TSDIR} | nproc {NPROC} | main {len(SIDS['main'])}")

    mon = GpuMon()
    # ---- 1. models on every GPU (as the notebook) ----
    mon.phase = "load"
    wpath = C.find_weight_file(C.ARMS[0]["file"])
    devs = [f"cuda:{g}" for g in range(ngpu)]
    models = {}; t = time.perf_counter()
    for d in devs:
        models[d], _ = C.load_model(wpath, C.ARMS[0]["arch"], 384, d)
    R["model_load_s"] = time.perf_counter() - t
    log(f"models loaded on {devs} in {R['model_load_s']:.1f}s")

    # ---- 2. warm-up study (prep + cudnn autotune V0) ----
    mon.phase = "warmup"
    ref = FR.make_builder("study_lut", C.IMG, C.SLOTS, hdr_threads=C.FR_HDR, px_threads=C.FR_PX)
    t = time.perf_counter(); wv, wm = ref(SIDS["warmup"][0], SER, TSDIR); R["warmup_prep_s"] = time.perf_counter() - t
    R["warmup_V0_s"] = warmup(v_notebook, models, wv, wm)
    log(f"warm-up prep {R['warmup_prep_s']:.2f}s, V0 autotune {R['warmup_V0_s']:.1f}s")

    # ---- 3. serial cold per-stage profile on 30 separate studies (cuda:0) ----
    mon.phase = "serial"
    ctx = FR._Ctx(FR.Cfg(hdr_threads=C.FR_HDR, px_threads=C.FR_PX, lut=True))
    d0 = devs[0]; rows = []; ident = []
    for i, sid in enumerate(SIDS["serial"]):
        T = Timer(); meta = {}
        t0 = time.perf_counter()
        vol, mask = build_lut_timed(sid, SER, TSDIR, ctx, T, meta)
        T.d["prep"] = time.perf_counter() - t0
        _sync(d0); t = time.perf_counter()
        xw = C.gpu_windows(vol, mask, C.K_EVAL, RES, d0)
        xm = C.gpu_windows(vol, mask, C.K_EVAL, RES, d0, norm=C.NORM, mirror=True)
        _sync(d0); T.add("windows", time.perf_counter() - t); t = time.perf_counter()
        C.infer_probs(models[d0], xw, d0); T.add("fwd_plain", time.perf_counter() - t); t = time.perf_counter()
        C.infer_probs(models[d0], xm, d0); T.add("fwd_mirror", time.perf_counter() - t)
        T.d["total"] = time.perf_counter() - t0
        if i < 3:                                         # byte identity vs the notebook's builder (warm)
            rv = ref(sid, SER, TSDIR)
            ident.append(bool(np.array_equal(rv[0], vol) and np.array_equal(rv[1], mask)))
        rows.append(dict(sid=sid, **T.d, **meta))
    df = pd.DataFrame(rows)
    R["serial"] = dict(n=len(df), identical=ident, mean_ms=(df.mean(numeric_only=True) * 1000).round(1).to_dict(),
                       median_ms=(df.median(numeric_only=True) * 1000).round(1).to_dict(),
                       mb_per_study=round(df.bytes.mean() / 1e6, 1),
                       px_read_MBps=round(df.bytes.sum() / 1e6 / df.px_read.sum(), 1),
                       nvsmi=mon.summary("serial"))
    for k in ("files", "picks_unique", "bytes"):
        R["serial"]["mean_ms"].pop(k, None); R["serial"]["median_ms"].pop(k, None)
    log("serial cold mean ms", R["serial"]["mean_ms"], "identical", ident)

    # ---- 4. pool prep throughput alone? (= V0 disk feed time below) ; V0 cold end-to-end ----
    main_sids = SIDS["main"]; stash = {}; runs = []
    runs.append(run_variant("V0", v_notebook, main_sids, models, mon, "disk", stash=stash))
    R["V0_cold"] = runs[-1]
    json.dump(R, open(f"{W}/exp026_results.json", "w"), indent=1, default=str)

    # ---- 5. warm variants: notebook pipeline from (page-cached) disk + GPU-only from RAM ----
    plan = [("V0", v_notebook, False), ("V1", make_tta(94), True), ("V2", v_notta, True),
            ("V3", v_sagtta, True), ("V4", make_tta(47), True), ("V0b", v_notebook, False)]
    R["warmup_variant_s"] = {}
    for name, fn, cl in plan:
        set_layout(models, cl)
        R["warmup_variant_s"][name] = warmup(fn, models, wv, wm)
        for src in ("disk", "ram"):
            runs.append(run_variant(name, fn, main_sids, models, mon, src, stash=stash))
        R["runs"] = runs
        json.dump(R, open(f"{W}/exp026_results.json", "w"), indent=1, default=str)
    set_layout(models, False)
    mon.stop()
    R["elapsed_s"] = time.time() - T_START
    json.dump(R, open(f"{W}/exp026_results.json", "w"), indent=1, default=str)
    log("DONE", round(R["elapsed_s"]), "s")
    for r in runs:
        log(f"  {r['variant']:4s} {r['source']:4s} {r['s_per_study']:.3f} s/study  steady {r['steady_s_per_study']:.3f}"
            f"  autotune {r['autotune_excess_s']:.1f}s")


if __name__ == "__main__":
    main()
