#!/usr/bin/env python3
"""exp041_t4_sin_slots -- Raptor inference without the Coronal non-fluid (3) and Axial (4) slots, Kaggle T4 x2.

Runs INSIDE a Kaggle kernel. The previous kernel cell writes (by make_kernel.py):
  /kaggle/working/raptor_cell.py   code cell of `nartaa/rsna-knee-0949-anatomical-mirror` (verbatim, 384 model)
  /kaggle/working/cell224.py       code cell of `nartaa/rsna-knee-0945-efficient-224crop` (verbatim, 224 model)
  /kaggle/working/exp041_sids.json {"main": [...150], "warmup": [...2], "cold5": [...150], "cold3": [...150]}
Nothing of the third-party code is copied here: it is imported and its functions are called.

Slot removal as in exp040 (missing series = zero slices, mask 0): prep runs the notebook's study_lut reader with
SLOTS[:3] (66 slices; the DICOMs of slots 3/4 are never listed nor read; series picks of slots 0-2 unchanged) and
the GPU consumer zero-pads the volume back to 96 slices. Windows come from _eval_centers on the mask: 64 centres
for a full 66-slice study -> K = 64 (one window per centre, same density as the 94 windows of C).

Variants (see SPEC.md):
  C    exp027 C: 384, crop 320, 94 windows, no TTA, model.half(), channels_last, chunks of 16, no cudnn.benchmark
  Ch   C with the 224 model in one batch of 94 + half (exp027 submit_E1_94)
  C3   C without slots 3/4 (3-slot read, K = 64)       C3h  C3 with the 224 model (one batch of 64, half)
  C3t  C3 + anatomical mirror TTA (64 + 64 windows)    Cb   C repeated at the end (drift control)
Outputs in /kaggle/working: exp041_results.json, preds_<V>_<src>.csv, gpu_util.csv
"""
import contextlib, copy, importlib.util, json, os, queue, subprocess, sys, threading, time, traceback
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
C2 = _imp("cell224", f"{W}/cell224.py")
SIDS = json.load(open(f"{W}/exp041_sids.json"))
_N = int(os.environ.get("EXP041_N", "0"))            # smoke only
if _N:
    SIDS = {k: v[:_N] for k, v in SIDS.items()}
LAB = C.LAB
F384, F224 = C.ARMS[0], C2.ARMS[0]
SLOTS3 = C.SLOTS[:3]                                     # Sag fluid 26, Sag non-fluid 22, Cor fluid 18
D3 = sum(s[2] for s in SLOTS3)                           # 66
K5, K3 = C.K_EVAL, 64                                    # 94 / 64 window centres for a full study


def _pre_init3():                                        # Pool initializer: 3-slot reader (no slot 3/4 I/O)
    C._READER = FR.make_builder(C.FASTREAD_VARIANT, C.IMG, SLOTS3, hdr_threads=C.FR_HDR, px_threads=C.FR_PX)


def _sync(d): torch.cuda.synchronize(d)


BASE = dict(model="384", chunk=16, half=True, slots=5, K=K5, tta=False)
VARS = {
    "C":   dict(BASE),
    "Ch":  dict(BASE, model="224", chunk=94),
    "C3":  dict(BASE, slots=3, K=K3),
    "C3h": dict(BASE, model="224", chunk=94, slots=3, K=K3),
    "C3t": dict(BASE, slots=3, K=K3, tta=True),
}


def pad96(vol, mask):
    if vol.shape[0] == C.MAXS:
        return vol, mask
    v = np.zeros((C.MAXS,) + vol.shape[1:], vol.dtype); v[:vol.shape[0]] = vol
    m = np.zeros(C.MAXS, mask.dtype); m[:mask.shape[0]] = mask
    return v, m


class Runner:
    """Per-device fp16 channels_last models for one variant: fn(vol, mask, d) -> (probs, t_windows, t_fwd)."""

    def __init__(self, name, cfg, base_models):
        self.name, self.cfg = name, cfg
        self.models = {}
        for d, m in base_models[cfg["model"]].items():
            self.models[d] = m                               # base models are already half + channels_last

    def windows(self, vol, mask, d, mirror=False):
        if self.cfg["model"] == "384":
            x = C.gpu_windows(vol, mask, self.cfg["K"], C.INPUT_RES, d, norm=C.NORM, mirror=mirror)
        else:
            assert not mirror
            x = C2.gpu_windows(vol, mask, self.cfg["K"], 224, d)
        return x.half()

    @torch.no_grad()
    def fwd(self, x, d):
        m = self.models[d]
        f = torch.cat([m.backbone(c.contiguous(memory_format=torch.channels_last))
                       for c in x.split(self.cfg["chunk"])], 0)
        logits = m.head(f.view(1, x.shape[0], -1)).float()
        assert torch.isfinite(logits).all(), "Nonfinite logits"
        return torch.sigmoid(logits)[0].cpu().numpy()

    def __call__(self, vol, mask, d):
        t = time.perf_counter()
        vol, mask = pad96(vol, mask)
        x = self.windows(vol, mask, d)
        xm = self.windows(vol, mask, d, mirror=True) if self.cfg["tta"] else None
        _sync(d); t1 = time.perf_counter()
        p = self.fwd(x, d)
        if xm is not None:
            p = 0.5 * (p + self.fwd(xm, d))                  # notebook's infer_pair_probs: mean of sigmoids
        return p, t1 - t, time.perf_counter() - t1


class GpuMon:
    def __init__(self):
        self.rows = []; self.phase = "init"
        try:
            self.p = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,clocks.sm,temperature.gpu",
                 "--format=csv,noheader,nounits", "-lms", "500"], stdout=subprocess.PIPE, text=True)
            threading.Thread(target=self._read, daemon=True).start()
        except Exception as e:
            log("nvidia-smi unavailable", e); self.p = None

    def _read(self):
        for line in self.p.stdout:
            try:
                i, u, m, clk, tmp = [x.strip() for x in line.split(",")]
                self.rows.append((time.time() - T_START, self.phase, int(i), float(u), float(m), float(clk), float(tmp)))
            except Exception:
                pass

    def summary(self, phase):
        # snapshot: the reader thread appends while we build the frame (lost the D ram run in kernel v1)
        df = pd.DataFrame(list(self.rows), columns=["t", "phase", "gpu", "util", "mem", "clk", "temp"])
        df = df[df.phase == phase]
        return {int(g): dict(util_mean=round(float(x.util.mean()), 1), mem_max=float(x.mem.max()),
                             clk_mean=round(float(x.clk.mean())), temp_max=float(x.temp.max()), n=int(len(x)))
                for g, x in df.groupby("gpu")}

    def stop(self):
        if self.p is not None: self.p.terminate()
        pd.DataFrame(self.rows, columns=["t", "phase", "gpu", "util", "mem", "clk", "temp"]).to_csv(
            f"{W}/gpu_util.csv", index=False)


# ---- pipeline runner (the notebook's main: fork Pool prep -> queue -> one thread per GPU) ------
def run_variant(name, fn, sids, devs, mon, source, stash=None, tag=None, init=None):
    """source='disk': notebook pipeline with Pool prep; source='ram': feed stashed volumes."""
    tag = tag or name
    q = queue.Queue(maxsize=C.QUEUE_MAX); probs = {}; lock = threading.Lock()
    st = {d: dict(n=0, busy=0.0, windows=0.0, fwd=0.0, wait=0.0, fail=0) for d in devs}
    per = {d: [] for d in devs}

    def consumer(d):
        while True:
            tw = time.perf_counter(); item = q.get(); st[d]["wait"] += time.perf_counter() - tw
            if item is None: q.task_done(); return
            sid, vol, mask = item
            t = time.perf_counter()
            try:
                p, a, b = fn(vol, mask, d)
                st[d]["windows"] += a; st[d]["fwd"] += b
            except Exception as e:
                log(f"  [{tag} {d}] {sid[:16]} FAIL {type(e).__name__}: {e}")
                p = np.full(len(LAB), 0.5, np.float32); st[d]["fail"] += 1
            bt = time.perf_counter() - t; per[d].append(bt)
            st[d]["busy"] += bt; st[d]["n"] += 1
            with lock:
                probs[sid] = np.asarray(p, np.float64)
                if stash is not None and source == "disk": stash[sid] = (vol, mask)
            q.task_done()

    mon.phase = f"{tag}_{source}"
    ths = [threading.Thread(target=consumer, args=(d,), daemon=True) for d in devs]
    t0 = time.perf_counter()
    for th in ths: th.start()
    if source == "disk":
        C._SER, C._TSDIR = SER, TSDIR
        with mp.get_context("fork").Pool(NPROC, initializer=init or C._pre_init) as pool:
            for sid, vol, mask in pool.imap_unordered(C._pre_one, sids, chunksize=2):
                q.put((sid, vol, mask))
    else:
        for sid in sids:
            q.put((sid,) + stash[sid])
    t_feed = time.perf_counter() - t0
    for _ in devs: q.put(None)
    q.join()
    for th in ths: th.join()
    wall = time.perf_counter() - t0
    n = len(sids)
    out = dict(variant=name, tag=tag, source=source, n=n, wall=wall, s_per_study=wall / n, t_feed=t_feed,
               per_gpu={d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()} for d, s in st.items()},
               nvsmi=mon.summary(mon.phase))
    for d, s in st.items():
        out["per_gpu"][d]["gpu_s_per_study"] = round(s["busy"] / max(1, s["n"]), 4)
        out["per_gpu"][d]["busy_frac"] = round(s["busy"] / wall, 3)
        if per[d]:
            med = float(np.median(per[d][1:])) if len(per[d]) > 1 else per[d][0]
            out["per_gpu"][d].update(first_study_s=round(per[d][0], 3), steady_gpu_s_per_study=round(med, 4))
    # cudnn.benchmark autotunes again in every new consumer thread (per-thread cache, exp026): steady wall =
    # wall minus the largest per-GPU excess of the first study (also catches any first-call compile in-thread).
    exc = max([out["per_gpu"][d].get("first_study_s", 0) - out["per_gpu"][d].get("steady_gpu_s_per_study", 0)
               for d in devs] + [0.0])
    out["autotune_excess_s"] = round(exc, 2)
    out["steady_s_per_study"] = (wall - exc) / n
    out["fails"] = int(sum(s["fail"] for s in st.values()))
    log(f"{tag:8s} {source:4s} wall {wall:7.1f}s  {wall / n:.3f} s/st  steady {out['steady_s_per_study']:.3f}  "
        f"feed {t_feed:6.1f}s  exc {exc:5.1f}s  " +
        "  ".join(f"{d}: n={s['n']} gpu={s['busy'] / max(1, s['n']):.3f}s/st wait={s['wait']:.0f}s fail={s['fail']}"
                  for d, s in st.items()))
    pd.DataFrame([[s] + list(probs[s]) for s in sids], columns=["StudyInstanceUID"] + LAB).to_csv(
        f"{W}/preds_{tag}_{source}.csv", index=False)
    return out


def warm(runner, devs, vol, mask, reps=2):
    """Main-thread warm-up / compile, per device in sequence; returns seconds of the first call per device."""
    first = {}
    for d in devs:
        t = time.perf_counter(); runner(vol, mask, d); _sync(d); first[d] = time.perf_counter() - t
        for _ in range(reps - 1): runner(vol, mask, d)
        _sync(d)
    return first


# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
def main():
    global SER, TSDIR, NPROC
    torch.backends.cudnn.benchmark = False                    # exp027 C
    R = dict(exp="exp041_t4_sin_slots", import_s=IMPORT_S)
    ngpu = torch.cuda.device_count()
    R["env"] = dict(torch=torch.__version__, timm=__import__("timm").__version__, cudnn=torch.backends.cudnn.version(),
                    cpu_count=os.cpu_count(), ngpu=ngpu, gpus=[torch.cuda.get_device_name(i) for i in range(ngpu)],
                    ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1))
    log("env", R["env"])
    ROOT = C.find_test_root()
    TSDIR = next(p for p in (ROOT + "/train_series", ROOT + "/train_images") if os.path.isdir(p))
    s = pd.read_csv(ROOT + "/train_series.csv")
    s["StudyInstanceUID"] = s["StudyInstanceUID"].astype(str); s["SeriesInstanceUID"] = s["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in s.groupby("StudyInstanceUID")}
    NPROC = max(4, min(96, (os.cpu_count() or 8) - 8))
    R["nproc"] = NPROC
    for k in SIDS:
        miss = [x for x in SIDS[k] if x not in SER]
        assert not miss, f"{k}: {len(miss)} studies missing from train_series.csv"
    log(f"root {ROOT} | nproc {NPROC} | " + " ".join(f"{k} {len(v)}" for k, v in SIDS.items()))

    mon = GpuMon()
    devs = [f"cuda:{g}" for g in range(ngpu)]
    mon.phase = "load"
    base = {"384": {}, "224": {}}; R["model_load_s"] = {}
    t = time.perf_counter()
    for d in devs:
        m, _ = C.load_model(C.find_weight_file(F384["file"]), F384["arch"], 384, d)
        base["384"][d] = m.half().to(memory_format=torch.channels_last)
    R["model_load_s"]["384"] = time.perf_counter() - t
    t = time.perf_counter()
    for d in devs:
        m, res = C2.load_model(C.find_weight_file(F224["file"]), F224["arch"], 224, d)
        base["224"][d] = m.half().to(memory_format=torch.channels_last)
    R["model_load_s"]["224"] = time.perf_counter() - t
    assert res == 224
    log(f"models loaded on {devs}: {R['model_load_s']}")

    ref5 = FR.make_builder("study_lut", C.IMG, C.SLOTS, hdr_threads=C.FR_HDR, px_threads=C.FR_PX)
    ref3 = FR.make_builder("study_lut", C.IMG, SLOTS3, hdr_threads=C.FR_HDR, px_threads=C.FR_PX)
    wv, wm = ref5(SIDS["warmup"][0], SER, TSDIR)
    w3v, w3m = ref3(SIDS["warmup"][0], SER, TSDIR)
    # the 3-slot read must equal the first 66 slices of the 5-slot read (same series picks, same pixels)
    R["slots3_identical_to_first66"] = bool(np.array_equal(w3v, wv[:D3]) and np.array_equal(w3m, wm[:D3]))
    log("3-slot volume == first 66 slices of the 5-slot volume:", R["slots3_identical_to_first66"])

    main_sids = SIDS["main"]; stash = {5: {}, 3: {}}; runs = []; R["runs"] = runs
    R["warm_first_s"] = {}; R["errors"] = {}; runners = {}

    def dump():
        json.dump(R, open(f"{W}/exp041_results.json", "w"), indent=1, default=str)

    def get_runner(v):
        if v not in runners:
            runners[v] = Runner(v, VARS[v], base)
            R["warm_first_s"][v] = warm(runners[v], devs, wv, wm)
            log(f"{v}: first call per GPU (main thread) {R['warm_first_s'][v]}")
        return runners[v]

    def run(v, src, sids=None, tag=None, keep=False):
        cfg = VARS[v]
        try:
            runs.append(run_variant(v, get_runner(v), sids or main_sids, devs, mon, src,
                                    stash=stash[cfg["slots"]], tag=tag or v,
                                    init=_pre_init3 if cfg["slots"] == 3 else None))
        except Exception as e:
            R["errors"][f"{tag or v}_{src}"] = f"{type(e).__name__}: {e}"; traceback.print_exc()
        dump()

    # ---- 1. C on the first disk read of the 150 main studies (stashes the 5-slot volumes) ----
    run("C", "disk", tag="C_first")
    # ---- 2. warm variants (C3 disk run stashes the 3-slot volumes before the RAM runs) ----
    for v in ["C", "Ch", "C3", "C3h", "C3t"]:
        for src in ("disk", "ram"):
            run(v, src)
    run("C", "disk", tag="Cb"); run("C", "ram", tag="Cb")
    # ---- 3. prep alone, 5 vs 3 slots, same (warm) studies ----
    noop = lambda vol, mask, d: (np.full(len(LAB), 0.5), 0.0, 0.0)
    runs.append(run_variant("prep5", noop, main_sids, devs, mon, "disk", tag="prep5_warm"))
    runs.append(run_variant("prep3", noop, main_sids, devs, mon, "disk", tag="prep3_warm", init=_pre_init3))
    dump()
    # ---- 4. cold: fastest 3-slot variant on cold3, its 5-slot twin on cold5 (different new studies) ----
    def cost(v):
        r = [x for x in runs if x["tag"] == v and x["source"] == "disk"]
        return r[0]["steady_s_per_study"] if r and not r[0]["fails"] else np.inf
    if SIDS.get("cold3") and SIDS.get("cold5"):
        best3 = min(("C3", "C3h"), key=cost); twin = {"C3": "C", "C3h": "Ch"}[best3]
        R["cold_pair"] = [best3, twin]
        run(best3, "disk", sids=SIDS["cold3"], tag=f"{best3}_cold")
        run(twin, "disk", sids=SIDS["cold5"], tag=f"{twin}_cold")
    mon.stop()
    R["elapsed_s"] = time.time() - T_START
    dump()
    log("DONE", round(R["elapsed_s"]), "s")
    for r in runs:
        log(f"  {r['tag']:10s} {r['source']:4s} {r['s_per_study']:.3f} s/st  steady {r['steady_s_per_study']:.3f}"
            f"  exc {r['autotune_excess_s']:.1f}s  feed {r['t_feed']:.1f}s  fails {r['fails']}")


if __name__ == "__main__":
    main()
