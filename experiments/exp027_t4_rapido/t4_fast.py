#!/usr/bin/env python3
"""exp027_t4_rapido -- faster T4 variants of the Raptor inference (no TTA), timed on Kaggle's real T4 x2.

Runs INSIDE a Kaggle kernel. The previous kernel cell writes (by make_kernel.py):
  /kaggle/working/raptor_cell.py   code cell of `nartaa/rsna-knee-0949-anatomical-mirror` (verbatim, 384 model)
  /kaggle/working/cell224.py       code cell of `nartaa/rsna-knee-0945-efficient-224crop` (verbatim, 224 model)
  /kaggle/working/exp027_sids.json {"main": [...150], "warmup": [...2], "cold": [...150]}
Nothing of the third-party code is copied here: it is imported and its functions are called.
Prep (study_lut volumes 96x384x384) is the 0949 notebook's for every variant; the 224 model gets its own
notebook's windows (centre crop 0.8 -> bilinear 224 -> ImageNet).

Variants (all without TTA, 94 windows, one pass) -- see SPEC.md:
  R    exp026 V2: 384 model, crop 320, channels_last, fp16 autocast, chunks of 16, cudnn.benchmark
  A    R without cudnn.benchmark
  B    A with the 224 model (chunks of 16)      B94  B with the 94 windows in one batch
  C    A with model.half() + half windows       D    A + torch.compile(backbone, max-autotune-no-cudagraphs)
  E1   B + C                                    E2   B + C + D        E2d  B + C + torch.compile default mode
  F    B + C through torch_tensorrt / onnxruntime-gpu if the image has it (never installed)
  Rb   R repeated (drift control); fastest of B/B94/E1 cold on 150 new fold-0 studies (first disk read);
  compiled variants run last (compile leaves CPU load behind) and E1b repeats E1 after them (control)

Outputs in /kaggle/working: exp027_results.json, preds_<V>_<src>.csv, gpu_util.csv
"""
import copy, contextlib, importlib.util, json, os, queue, subprocess, sys, threading, time, traceback
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
IMPORT_S = time.time() - t                               # notebook-equivalent import cost (torch + timm + cell)
C2 = _imp("cell224", f"{W}/cell224.py")                  # second cell: defs only (shares the torch import)
SIDS = json.load(open(f"{W}/exp027_sids.json"))
_N = int(os.environ.get("EXP027_N", "0"))            # smoke only: cut the study lists
if _N:
    SIDS = {k: v[:_N] for k, v in SIDS.items()}
LAB = C.LAB
F384 = C.ARMS[0]
F224 = C2.ARMS[0]
K = C.K_EVAL                                             # 94 in both notebooks
assert C2.K_EVAL == K


def _sync(d): torch.cuda.synchronize(d)


# ---- variant configs ------------------------------------------------------------------------
BASE = dict(model="384", chunk=16, cl=True, half=False, compile=None, bench=True, rt=None)
VARS = {
    "R":   dict(BASE),
    "A":   dict(BASE, bench=False),
    "B":   dict(BASE, bench=False, model="224"),
    "B94": dict(BASE, bench=False, model="224", chunk=94),
    "C":   dict(BASE, bench=False, half=True),
    "D":   dict(BASE, bench=False, compile="max-autotune-no-cudagraphs"),
    "E1":  dict(BASE, bench=False, model="224", half=True),
    "E2":  dict(BASE, bench=False, model="224", half=True, compile="max-autotune-no-cudagraphs"),
    "E2d": dict(BASE, bench=False, model="224", half=True, compile="default"),
    "F":   dict(BASE, bench=False, model="224", half=True, chunk=94, rt="auto"),
}


class Runner:
    """Holds per-device model objects for one variant: fn(vol, mask, d) -> (probs, t_windows, t_fwd)."""

    def __init__(self, name, cfg, base_models):
        self.name, self.cfg = name, cfg
        self.models, self.bbs = {}, {}
        for d, m in base_models[cfg["model"]].items():
            m = copy.deepcopy(m) if cfg["half"] else m
            if cfg["half"]:
                m.half()
            m.to(memory_format=torch.channels_last if cfg["cl"] else torch.contiguous_format)
            self.models[d] = m
            self.bbs[d] = m.backbone
        if cfg["compile"]:
            for d, m in self.models.items():
                self.bbs[d] = torch.compile(m.backbone, mode=cfg["compile"], dynamic=False)
        self.rt = None
        if cfg["rt"]:
            self.rt = build_rt(self.models, cfg)              # raises if no runtime is available

    def windows(self, vol, mask, d):
        if self.cfg["model"] == "384":
            x = C.gpu_windows(vol, mask, K, C.INPUT_RES, d)
        else:
            x = C2.gpu_windows(vol, mask, K, 224, d)
        return x.half() if self.cfg["half"] else x

    @torch.no_grad()
    def fwd(self, x, d):
        cfg, m = self.cfg, self.models[d]
        if self.rt is not None:
            f = self.rt(x, d)                                 # (K, F) half features
        else:
            ctx = contextlib.nullcontext() if cfg["half"] else torch.autocast("cuda", dtype=torch.float16)
            with ctx:
                fmt = torch.channels_last if cfg["cl"] else torch.contiguous_format
                f = torch.cat([self.bbs[d](c.contiguous(memory_format=fmt)) for c in x.split(cfg["chunk"])], 0)
        # head as in the notebook: inside autocast (fp32 master) or all-fp16 (model.half()); fp32 logits
        ctx = contextlib.nullcontext() if cfg["half"] else torch.autocast("cuda", dtype=torch.float16)
        with ctx:
            logits = m.head(f.view(1, x.shape[0], -1)).float()
        assert torch.isfinite(logits).all(), "Nonfinite logits"
        return torch.sigmoid(logits)[0].cpu().numpy()

    def __call__(self, vol, mask, d):
        t = time.perf_counter()
        x = self.windows(vol, mask, d)
        _sync(d); t1 = time.perf_counter()
        p = self.fwd(x, d)
        return p, t1 - t, time.perf_counter() - t1


def build_rt(models, cfg):
    """F: backbone through torch_tensorrt (fp16) or onnxruntime-gpu (CUDA / TensorRT EP). Never installs."""
    info = {}
    try:
        import torch_tensorrt  # noqa
        info["torch_tensorrt"] = torch_tensorrt.__version__
    except Exception as e:
        info["torch_tensorrt"] = f"absent ({type(e).__name__})"
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__; info["ort_providers"] = ort.get_available_providers()
    except Exception as e:
        ort = None; info["onnxruntime"] = f"absent ({type(e).__name__})"
    RT_INFO.update(info)
    if not str(info["torch_tensorrt"]).startswith("absent"):
        import torch_tensorrt
        trt = {}
        for d, m in models.items():
            ex = torch.randn(K, 3, 224, 224, device=d).half().contiguous(memory_format=torch.channels_last)
            trt[d] = torch_tensorrt.compile(m.backbone, ir="dynamo", inputs=[ex], enabled_precisions={torch.half})
        RT_INFO["used"] = "torch_tensorrt"
        return lambda x, d: trt[d](x.contiguous(memory_format=torch.channels_last))
    if ort is not None and ({"CUDAExecutionProvider", "TensorrtExecutionProvider"} & set(info["ort_providers"])):
        d0 = next(iter(models)); p = f"{W}/bb224_half.onnx"
        ex = torch.randn(K, 3, 224, 224, device=d0).half()
        torch.onnx.export(models[d0].backbone, (ex,), p, input_names=["x"], output_names=["f"], opset_version=17,
                          dynamo=False)
        sess = {}
        for d in models:
            i = int(str(d).split(":")[1])
            prov = [("CUDAExecutionProvider", {"device_id": i})]
            sess[d] = ort.InferenceSession(p, providers=prov)
        RT_INFO["used"] = "onnxruntime CUDA EP"

        def run(x, d):
            o = sess[d].run(None, {"x": x.contiguous().cpu().numpy()})[0]
            return torch.from_numpy(o).to(d)
        return run
    raise RuntimeError(f"no TensorRT / ORT-GPU runtime in the image: {info}")


RT_INFO = {}


# ---- nvidia-smi sampler -----------------------------------------------------------------------
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
def run_variant(name, fn, sids, devs, mon, source, stash=None, tag=None):
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
        with mp.get_context("fork").Pool(NPROC, initializer=C._pre_init) as pool:
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
def main():
    global SER, TSDIR, NPROC
    R = dict(exp="exp027_t4_rapido", import_s=IMPORT_S)
    ngpu = torch.cuda.device_count()
    env = dict(torch=torch.__version__, timm=__import__("timm").__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), cpu_count=os.cpu_count(),
               affinity=len(os.sched_getaffinity(0)), ngpu=ngpu, gpus=[torch.cuda.get_device_name(i) for i in range(ngpu)],
               ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1))
    for mod in ("triton", "torch_tensorrt", "tensorrt", "onnxruntime", "onnx", "onnxscript"):
        try:
            m = importlib.import_module(mod); env[mod] = getattr(m, "__version__", "?")
        except Exception as e:
            env[mod] = f"absent ({type(e).__name__})"
    try:
        import onnxruntime as ort; env["ort_providers"] = ort.get_available_providers()
    except Exception:
        pass
    R["env"] = env
    log("env", env)
    ROOT = C.find_test_root()
    TSDIR = next(p for p in (ROOT + "/train_series", ROOT + "/train_images") if os.path.isdir(p))
    s = pd.read_csv(ROOT + "/train_series.csv")
    s["StudyInstanceUID"] = s["StudyInstanceUID"].astype(str); s["SeriesInstanceUID"] = s["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in s.groupby("StudyInstanceUID")}
    NPROC = max(4, min(96, (os.cpu_count() or 8) - 8))          # notebook's formula
    R["nproc"] = NPROC
    for k in SIDS:
        miss = [x for x in SIDS[k] if x not in SER]
        assert not miss, f"{k}: {len(miss)} studies missing from train_series.csv"
    log(f"root {ROOT} | tsdir {TSDIR} | nproc {NPROC} | " + " ".join(f"{k} {len(v)}" for k, v in SIDS.items()))

    mon = GpuMon()
    devs = [f"cuda:{g}" for g in range(ngpu)]
    # ---- 1. both models on every GPU (fp32 masters, as the notebooks load them) ----
    mon.phase = "load"
    base = {"384": {}, "224": {}}; R["model_load_s"] = {}
    t = time.perf_counter()
    for d in devs:
        base["384"][d], _ = C.load_model(C.find_weight_file(F384["file"]), F384["arch"], 384, d)
    R["model_load_s"]["384"] = time.perf_counter() - t
    t = time.perf_counter()
    for d in devs:
        base["224"][d], res = C2.load_model(C.find_weight_file(F224["file"]), F224["arch"], 224, d)
    R["model_load_s"]["224"] = time.perf_counter() - t
    R["res224_ckpt"] = res
    assert res == 224, res
    log(f"models loaded on {devs}: {R['model_load_s']}")

    ref = FR.make_builder("study_lut", C.IMG, C.SLOTS, hdr_threads=C.FR_HDR, px_threads=C.FR_PX)
    wv, wm = ref(SIDS["warmup"][0], SER, TSDIR)

    main_sids = SIDS["main"]; stash = {}; runs = []; R["runs"] = runs
    R["warm_first_s"] = {}; R["build_s"] = {}; R["errors"] = {}
    runners = {}

    def dump():
        R["rt_info"] = RT_INFO
        json.dump(R, open(f"{W}/exp027_results.json", "w"), indent=1, default=str)

    def get_runner(v):
        if v not in runners:
            cfg = VARS[v]
            torch.backends.cudnn.benchmark = cfg["bench"]
            t = time.perf_counter()
            runners[v] = Runner(v, cfg, base)
            R["build_s"][v] = time.perf_counter() - t
            R["warm_first_s"][v] = warm(runners[v], devs, wv, wm)     # compile happens here for D / E2 / F
            if cfg["compile"]:
                # inductor leaves a compile-worker pool alive: it slowed every later run in the smoke (CPU)
                try:
                    from torch._inductor import async_compile
                    async_compile.shutdown_compile_workers()
                except Exception as e:
                    log("shutdown_compile_workers failed", e)
            log(f"{v}: built {R['build_s'][v]:.1f}s, first call per GPU (main thread) {R['warm_first_s'][v]}")
        torch.backends.cudnn.benchmark = VARS[v]["bench"]
        return runners[v]

    # ---- 2. R cold (first disk read of the 150 main studies) ----
    runs.append(run_variant("R", get_runner("R"), main_sids, devs, mon, "disk", stash=stash, tag="R_cold"))
    dump()

    def run_warm(v, name):
        try:
            fn = get_runner(name)
        except Exception as e:
            R["errors"][v] = f"{type(e).__name__}: {e}"
            log(f"{v}: build FAILED {type(e).__name__}: {e}"); traceback.print_exc()
            dump(); return
        for src in ("disk", "ram"):
            try:
                torch.backends.cudnn.benchmark = VARS[name]["bench"]
                runs.append(run_variant(name, fn, main_sids, devs, mon, src, stash=stash, tag=v))
            except Exception as e:
                R["errors"][f"{v}_{src}"] = f"{type(e).__name__}: {e}"; traceback.print_exc()
        dump()

    # ---- 3. warm variants without compile: disk (page cache) + RAM; Rb = R again (drift control) ----
    for v in ["R", "A", "B", "B94", "C", "E1", "Rb"]:
        run_warm(v, "R" if v == "Rb" else v)

    # ---- 4. fastest non-compiled 224 variant cold on 150 new studies (first disk read) + warm prep alone ----
    def cost(v):
        r = [x for x in runs if x["tag"] == v and x["source"] == "disk"]
        if not r or r[0]["fails"]: return np.inf
        return r[0]["steady_s_per_study"] * 1300 + r[0]["autotune_excess_s"]
    cands = [v for v in ("B", "B94", "E1") if np.isfinite(cost(v))]
    if cands and SIDS.get("cold"):
        best = min(cands, key=cost); R["cold_variant"] = best
        runs.append(run_variant(best, get_runner(best), SIDS["cold"], devs, mon, "disk", tag=f"{best}_cold"))
        runs.append(run_variant("prep", lambda vol, mask, d: (np.full(len(LAB), 0.5), 0.0, 0.0),
                                SIDS["cold"], devs, mon, "disk", tag="prep_warm"))
        dump()

    # ---- 5. compiled / external-runtime variants last (compile may leave CPU load behind); E1b = control ----
    for v in ["D", "E2", "E2d", "F", "E1b"]:
        run_warm(v, "E1" if v == "E1b" else v)

    mon.stop()
    R["elapsed_s"] = time.time() - T_START
    dump()
    log("DONE", round(R["elapsed_s"]), "s")
    for r in runs:
        log(f"  {r['tag']:8s} {r['source']:4s} {r['s_per_study']:.3f} s/st  steady {r['steady_s_per_study']:.3f}"
            f"  exc {r['autotune_excess_s']:.1f}s  feed {r['t_feed']:.1f}s  fails {r['fails']}")


if __name__ == "__main__":
    main()
