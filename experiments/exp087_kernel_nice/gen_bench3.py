"""[exp084, from eff2/gen_bench.py] Build a private Kaggle BENCH notebook that runs a scored inference kernel unchanged on 1300 TRAIN
studies (58 gold + 871 fold 0 + 371 random), on Kaggle T4x2 hardware, with timing instrumentation.
usage: python gen_bench.py <src.ipynb> <out_dir> <slug> <dataset_source> <variant>
variant: x (X1-X4) | xb (X1-X5) | xn (X1-X4 + X6 nice 10, exp087), see fastx.py; src = exp080-ctrl224-fast.ipynb"""
import json, sys, os, random
import pandas as pd

src, out_dir, slug, dsrc, variant = sys.argv[1:6]
REPO = r"C:/Users/medro/Documents/_PERSONAL_PROJECTS/RSNA_Challenge"
nb = json.load(open(src, encoding="utf-8"))
codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
assert len(codes) == 2
hashcell, body = codes

s = pd.read_csv(REPO + "/SPLITS/split_v1_seed42.csv")
gold = s[s.fold == -1].StudyInstanceUID.astype(str).tolist()
f0 = s[s.fold == 0].StudyInstanceUID.astype(str).tolist()
rest = sorted(s[s.fold.isin([1, 2, 3, 4])].StudyInstanceUID.astype(str).tolist())
rng = random.Random(0)
ids = gold + f0 + rng.sample(rest, 1300 - len(gold) - len(f0))
rng.shuffle(ids)
assert len(ids) == 1300 and len(set(ids)) == 1300


def rep(t, a, b, n=1):
    assert t.count(a) == n, (a[:60], t.count(a))
    return t.replace(a, b)


body = rep(body, 'if __name__ == "__main__":\n    main()', '')
# --- instrumentation (does not touch the math) ---
body = rep(body, "            item = q.get()\n",
           "            _w0 = time.time(); item = q.get(); GPU_WAIT[d] = GPU_WAIT.get(d, 0.0) + time.time() - _w0\n")
body = rep(body, "                        arm_probs[a][i] = infer_probs(model, xw, d)",
           "                        _g0 = time.time(); arm_probs[a][i] = infer_probs(model, xw, d); GPU_T[sid] = time.time() - _g0")
body = rep(body, "    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)",
           "    RAW = pd.DataFrame(arm_probs[0], columns=LAB); RAW.insert(0, 'StudyInstanceUID', test_ids); RAW.to_csv('/kaggle/working/raw_probs.csv', index=False)\n"
           "    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)")
body = rep(body, "    print(f\"DONE {time.time()-t0:.0f}s\", flush=True)",
           "    print(f\"DONE {time.time()-t0:.0f}s\", flush=True)\n    TIMES['main_total_s'] = time.time() - t0")
body = rep(body, "    t_pre = time.time()\n",
           "    t_pre = time.time(); TIMES['t_models_loaded'] = t_pre - t0\n")
body = rep(body, "    for _ in devs: q.put(None)\n",
           "    TIMES['pre_done'] = time.time() - t0; TIMES['pre_wall'] = time.time() - t_pre\n    for _ in devs: q.put(None)\n")
body = rep(body, "    print(f\"[gpu] done: {done_n[0]}/{N} studies",
           "    TIMES['gpu_done'] = time.time() - t0\n    print(f\"[gpu] done: {done_n[0]}/{N} studies")

FAST = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastx
early, body = fastx.apply(body, "FAKE", cudnn_bench=variant.endswith("b"), bench=True, nice=10 if variant.startswith("xn") else 0)
setup = f'''# BENCH ({variant}): unchanged scored pipeline on 1300 TRAIN studies; NOT a submission kernel.
import os, time, json, glob
T_NB0 = time.time()
{FAST}IDS = {json.dumps(ids)}
import pandas as pd
def _troot():
    for b in ["/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"]:
        if os.path.exists(b + "/train.csv"): return b
    for d, _, f in os.walk("/kaggle/input"):
        if "train_series.csv" in f: return d
TR = _troot(); print("train root", TR, flush=True)
FAKE = "/tmp/fakeroot"; os.makedirs(FAKE, exist_ok=True)
ts = pd.read_csv(TR + "/train_series.csv"); ts["StudyInstanceUID"] = ts["StudyInstanceUID"].astype(str)
ts[ts.StudyInstanceUID.isin(set(IDS))].to_csv(FAKE + "/test_series.csv", index=False)
pd.DataFrame({{"StudyInstanceUID": IDS}}).to_csv(FAKE + "/test.csv", index=False)
if not os.path.exists(FAKE + "/test_series"): os.symlink(TR + "/train_series", FAKE + "/test_series")
PRE_T, GPU_T, GPU_WAIT, QS, TIMES = {{}}, {{}}, {{}}, [], {{}}
'''
harness = r'''
_orig_find = find_test_root
def find_test_root():
    return FAKE
_tm = time.time(); main(); TIMES['main_wall'] = time.time() - _tm
import numpy as np
pre = np.array([PRE_T[k] for k in IDS if k in PRE_T]); g = np.array(list(GPU_T.values()))
rep_ = dict(variant=VARIANT, n=len(IDS), nproc_cpu=os.cpu_count(), times=TIMES, gpu_wait=GPU_WAIT,
            pre_wall_mean=float(pre[:, 0].mean()), pre_wall_p50=float(np.median(pre[:, 0])), pre_wall_p95=float(np.percentile(pre[:, 0], 95)),
            pre_cpu_mean=float(pre[:, 1].mean()), gpu_mean=float(g.mean()), gpu_p95=float(np.percentile(g, 95)),
            q_mean=float(np.mean(QS)), q_max=int(np.max(QS)), nb_elapsed_to_end=time.time() - T_NB0)
tr = pd.read_csv(_troot() + '/train.csv', encoding='utf-8')
tr['StudyInstanceUID'] = tr['StudyInstanceUID'].astype(str)
raw = pd.read_csv('/kaggle/working/raw_probs.csv'); raw['StudyInstanceUID'] = raw['StudyInstanceUID'].astype(str)
g_ = tr.dropna(subset=LAB).merge(raw, on='StudyInstanceUID', suffixes=('_y', ''))
from sklearn.metrics import roc_auc_score
aucs = [roc_auc_score(g_[c + '_y'].round().astype(int), g_[c]) for c in LAB if g_[c + '_y'].round().nunique() == 2]
rep_['early_start_s'] = T_EARLY0 - T_NB0; rep_['gold_n'] = len(g_); rep_['gold_auc'] = float(np.mean(aucs))
try:   # isolated GPU microbench (no prep load): eager vs CUDA graph, same input
    torch.cuda.synchronize()
    m, r = load_model(find_weight_file(ARMS[0]['file']), ARMS[0]['arch'], ARMS[0]['res'], 'cuda:0')
    xw = torch.randn(K_EVAL, 3, r, r, device='cuda:0')
    def _tm(fn, n=30):
        fn(); torch.cuda.synchronize(); t = time.time()
        for _ in range(n): o = fn()
        torch.cuda.synchronize(); return (time.time() - t) / n, o
    te, oe = _tm(lambda: infer_probs(m, xw, 'cuda:0'))
    rep_['micro_eager_s'] = te
    if '_Graph' in globals():
        G = _Graph(m, 'cuda:0', K_EVAL, r); tg, og = _tm(lambda: G(xw))
        rep_['micro_graph_s'] = tg; rep_['micro_graph_maxdiff'] = float(np.abs(oe - og).max())
except Exception as e:
    rep_['micro_err'] = repr(e)
print('BENCH_REPORT', json.dumps(rep_), flush=True)
json.dump(rep_, open('/kaggle/working/bench_report.json', 'w'), indent=1)
pd.DataFrame([(k,) + tuple(v) for k, v in PRE_T.items()], columns=['sid', 'pre_wall', 'pre_cpu']).assign(gpu=lambda d: d.sid.map(GPU_T)).to_csv('/kaggle/working/per_study.csv', index=False)
import shutil; shutil.rmtree(FAKE, ignore_errors=True)
'''
setup = setup.replace("PRE_T, GPU_T", f"VARIANT = {variant!r}\nPRE_T, GPU_T")
cells = [setup, early, hashcell, body, harness]
nbo = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c} for c in cells],
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
       "nbformat": 4, "nbformat_minor": 4}
os.makedirs(out_dir, exist_ok=True)
json.dump(nbo, open(f"{out_dir}/{slug}.ipynb", "w", encoding="utf-8"))
meta = {"id": f"medellinrodrigo/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [dsrc], "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
json.dump(meta, open(f"{out_dir}/kernel-metadata.json", "w"), indent=1)
print("ok", out_dir, len(ids))
