"""exp040 - robustez (solo inferencia) del modelo Raptor fold 0 de exp029.

Script auxiliar declarado en la SPEC: no entrena. Carga best_f0.pt, cachea los 929 estudios (val fold 0 + gold) en el
disco local y corre cada condicion de perturbacion. Guarda results/exp040_robustez/preds/<cond>.parquet
(index StudyInstanceUID, 12 columnas de probabilidad) y timings.csv.

Uso (desde la raiz del repo en el pod):
  RSNA_USER=rodrigo python experiments/exp040_robustez/infer_robustez.py [--smoke] [--only c1,c2]
"""
import argparse, json, math, sys, time, zlib
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.knee.config import LABELS
from src.knee.splits import load_split

CORPUS = "/workspace/data/cache/raptor96_384"
CKPT = "/workspace/users/rodrigo/wt_exp029/results/exp029_raptor_entreno/best_f0.pt"
CACHE = "/root/exp040_cache.npy"
OUT = REPO / "results" / "exp040_robustez" / "preds"
IMG, RES, K_EVAL = 384, 320, 24
SLOT_SIZES = [26, 22, 18, 12, 18]
SAG_SLOTS = (0, 1)
DEV = torch.device("cuda")
_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


# ---------------- modelo (verbatim exp029) ----------------
class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        self.n = n

    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    bb = timm.create_model(ck["arch"], pretrained=False, num_classes=0, in_chans=3, global_pool="avg", img_size=RES)
    m = RaptorClassifier(bb, F_dim=bb.num_features)
    m.load_state_dict(ck["model"])
    print(f"ckpt epoch {ck.get('epoch')} pseudo {ck.get('pseudo_auc')}", flush=True)
    return m.to(DEV).eval()


@torch.no_grad()
def forward_bags(model, bags, chunk=96):
    """bags: lista de tensores (k_i,3,RES,RES) normalizados -> (len, 12) probabilidades."""
    flat = torch.cat(bags, 0)
    feats = []
    for i in range(0, len(flat), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats.append(model.backbone(flat[i:i + chunk]).float())
    feats = torch.cat(feats, 0)
    out, o = [], 0
    for b in bags:
        f = feats[o:o + len(b)][None]; o += len(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model.head(f).float())[0])
    return torch.stack(out).cpu().numpy()


# ---------------- ventanas (= StudyWindows eval de exp029) ----------------
def eval_centers(mask, k):
    D = len(mask)
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    if k is None:          # todas las ventanas
        sel = cs
    else:
        idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
        sel = [cs[i] for i in idx]
    return np.array([max(1, min(c, D - 2)) for c in sel])


def slot_of(sizes):
    return np.repeat(np.arange(len(sizes)), sizes)


# ---------------- perturbaciones ----------------
def rng_for(uid, cond):
    return torch.Generator(device=DEV).manual_seed(zlib.crc32(f"{uid}|{cond}".encode()) & 0x7FFFFFFF)


def slice_plan(cond):
    """Devuelve (indices de corte en el volumen de 96, tamanos de slot nuevos) o None si no cambia."""
    b = np.cumsum([0] + SLOT_SIZES)
    if cond.startswith("cortes"):
        frac = {"cortes50": 0.5, "cortes25": 0.25}[cond]
        idx, sizes = [], []
        for s, k in enumerate(SLOT_SIZES):
            n = max(3, int(math.ceil(k * frac)))
            idx += list(b[s] + np.linspace(0, k - 1, n).round().astype(int)); sizes.append(n)
        return np.array(idx), sizes
    if cond.startswith("shift"):
        d = int(cond[5:])
        idx = []
        for s, k in enumerate(SLOT_SIZES):
            idx += list(np.clip(b[s] + np.arange(k) + d, b[s], b[s + 1] - 1))
        return np.array(idx), list(SLOT_SIZES)
    return None


def drop_slots(cond):
    return {"drop_s0": [0], "drop_s1": [1], "drop_s2": [2], "drop_s3": [3], "drop_s4": [4],
            "drop_cor": [2, 3], "drop_sag": [0, 1]}.get(cond, [])


def bias_field(g):
    coarse = torch.randn(1, 1, 4, 4, generator=g, device=DEV)
    f = F.interpolate(coarse, size=(IMG, IMG), mode="bicubic", align_corners=True)[0, 0]
    f = f / f.abs().max().clamp_min(1e-6)
    return torch.exp(0.2 * f)


def intensity(v, valid, cond, g):
    """v: (D,IMG,IMG) float [0,1] en GPU; valid: (D,) bool. Devuelve v perturbado."""
    if cond in ("ref", "mirror") or not any(cond.startswith(p) for p in ("gauss", "rician", "gamma", "contr", "bias", "res192")):
        return v
    x = v[valid]
    if cond.startswith("gauss"):
        s = float(cond[5:]) / 100
        x = x + s * torch.randn(x.shape, generator=g, device=DEV)
    elif cond.startswith("rician"):
        s = float(cond[6:]) / 100
        n1 = torch.randn(x.shape, generator=g, device=DEV); n2 = torch.randn(x.shape, generator=g, device=DEV)
        x = torch.sqrt((x + s * n1) ** 2 + (s * n2) ** 2)
    elif cond.startswith("gamma"):
        x = x.clamp_min(0) ** (float(cond[5:]) / 100)
    elif cond.startswith("contr"):
        c = 1 + float(cond[5:]) / 100
        m = x.mean()
        x = (x - m) * c + m
    elif cond == "bias":
        x = x * bias_field(g)[None]
    elif cond == "res192":
        x = F.interpolate(x[:, None], size=(IMG // 2, IMG // 2), mode="area")
        x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)[:, 0]
    v = v.clone()
    v[valid] = x.clamp(0, 1)
    return v


def crop_offsets(cond):
    lo = (IMG - RES) // 2
    dy = dx = 0
    if cond.startswith("cropx"):
        dx = int(cond[5:])
    elif cond.startswith("cropy"):
        dy = int(cond[5:])
    return lo + dy, lo + dx


def k_for(cond):
    return {"k6": 6, "k12": 12, "k48": 48, "kall": None}.get(cond, K_EVAL)


def make_bag(vol_u8, mask, uid, cond):
    """vol_u8: (96,IMG,IMG) uint8 tensor en GPU; mask: (96,) np. -> (k,3,RES,RES) normalizado."""
    sizes = list(SLOT_SIZES)
    plan = slice_plan(cond)
    vol = vol_u8
    mask = mask.copy()
    if plan is not None:
        idx, sizes = plan
        vol = vol_u8[torch.as_tensor(idx, device=DEV)]
        mask = mask[idx]
    for s in drop_slots(cond):
        b = np.cumsum([0] + sizes)
        mask[b[s]:b[s + 1]] = 0
    D = len(mask)
    cs = eval_centers(mask, k_for(cond))
    need = np.unique(np.concatenate([cs - 1, cs, cs + 1]))
    remap = -np.ones(D, np.int64); remap[need] = np.arange(len(need))
    v = vol[torch.as_tensor(need, device=DEV)].float() / 255.0
    valid_need = torch.as_tensor(mask[need] > 0, device=DEV)
    v = v * valid_need.view(-1, 1, 1)            # series quitadas -> cortes a cero
    v = intensity(v, valid_need, cond, rng_for(uid, cond))
    tri = v[torch.as_tensor(np.stack([remap[cs - 1], remap[cs], remap[cs + 1]], 1), device=DEV)]   # (k,3,IMG,IMG)
    if cond == "mirror":
        sag = torch.as_tensor(np.isin(slot_of(sizes)[cs], SAG_SLOTS), device=DEV).view(-1, 1, 1, 1)
        tri = torch.where(sag, tri.flip(1), tri.flip(-1))
    y0, x0 = crop_offsets(cond)
    tri = tri[:, :, y0:y0 + RES, x0:x0 + RES]
    return (tri - _MEAN) / _STD


CONDS = (["ref", "mirror", "k6", "k12", "k48", "kall",
          "gauss2", "gauss5", "gauss10", "rician5", "rician10",
          "gamma70", "gamma140", "contr+20", "contr-20", "bias",
          "cortes50", "cortes25", "shift+2", "shift-2",
          "drop_s0", "drop_s1", "drop_s2", "drop_s3", "drop_s4", "drop_cor", "drop_sag",
          "res192", "cropx+16", "cropx-16", "cropy+16", "cropy-16"])
PRIORITY = ["ref", "mirror", "cortes50", "cortes25", "shift+2", "shift-2", "drop_s0", "drop_s1", "drop_s2", "drop_s3",
            "drop_s4", "drop_cor", "drop_sag", "gauss2", "gauss5", "gauss10", "rician5", "rician10", "gamma70",
            "gamma140", "contr+20", "contr-20", "bias", "k12", "k6", "k48", "kall", "res192", "cropx+16", "cropx-16",
            "cropy+16", "cropy-16"]
assert sorted(PRIORITY) == sorted(CONDS)


def study_ids():
    split = load_split("split_v1_seed42")
    va = list(split.loc[(split.fold == 0), "StudyInstanceUID"].astype(str))
    gold = list(split.loc[split.is_gold == 1, "StudyInstanceUID"].astype(str))
    return va, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    va, gold = study_ids()
    ids_all = va + gold
    cids = np.load(f"{CORPUS}/ids.npy").astype(str)
    masks = np.load(f"{CORPUS}/masks.npy")
    id2row = {u: i for i, u in enumerate(cids)}
    ids_all = [u for u in ids_all if u in id2row]
    print(f"val {len(va)} gold {len(gold)} en corpus {len(ids_all)}", flush=True)
    rows = np.array([id2row[u] for u in ids_all])
    conds = [c for c in PRIORITY if not a.only or c in a.only.split(",")]
    if a.smoke:
        ids_all, rows = ids_all[:20], rows[:20]
        conds = conds[:3] if not a.only else conds
        cache = np.load(f"{CORPUS}/vols.npy", mmap_mode="r")
        local = {i: np.asarray(cache[r]) for i, r in enumerate(rows)}
        get = lambda i: local[i]
    else:
        meta_p = Path(CACHE + ".json")
        if not (Path(CACHE).exists() and meta_p.exists() and json.loads(meta_p.read_text())["ids"] == ids_all):
            t = time.time()
            src = np.load(f"{CORPUS}/vols.npy", mmap_mode="r")
            dst = np.lib.format.open_memmap(CACHE + ".tmp", mode="w+", dtype=np.uint8, shape=(len(rows), 96, IMG, IMG))
            order = np.argsort(rows)
            for j, i in enumerate(order):
                dst[i] = src[rows[i]]
                if j % 100 == 0:
                    print(f"  cache {j}/{len(rows)} {time.time() - t:.0f}s", flush=True)
            dst.flush(); del dst
            Path(CACHE + ".tmp").rename(CACHE)
            meta_p.write_text(json.dumps({"ids": ids_all}))
            print(f"cache {CACHE} en {time.time() - t:.0f}s", flush=True)
        arr = np.load(CACHE, mmap_mode="r")   # disco local; la page cache lo deja en RAM tras la 1.a pasada
        get = lambda i: arr[i]
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model()
    tim_p = OUT.parent / "timings.csv"
    timings = pd.read_csv(tim_p).to_dict("records") if tim_p.exists() and not a.smoke else []
    for cond in conds:
        dst = OUT / (f"smoke_{cond}.parquet" if a.smoke else f"{cond}.parquet")
        if dst.exists() and not a.smoke:
            print(f"skip {cond} (existe)", flush=True); continue
        t = time.time(); torch.cuda.synchronize()
        preds = np.zeros((len(ids_all), len(LABELS)), np.float32); nwin = 0
        for s in range(0, len(ids_all), a.bs):
            bags = []
            for i in range(s, min(s + a.bs, len(ids_all))):
                vol = torch.from_numpy(get(i)).to(DEV, non_blocking=True)
                bags.append(make_bag(vol, masks[rows[i]], ids_all[i], cond))
            nwin += sum(len(b) for b in bags)
            preds[s:s + len(bags)] = forward_bags(model, bags)
        torch.cuda.synchronize(); dt = time.time() - t
        pd.DataFrame(preds, index=pd.Index(ids_all, name="StudyInstanceUID"), columns=LABELS).to_parquet(dst)
        timings.append({"cond": cond, "seconds": dt, "n": len(ids_all), "windows_mean": nwin / len(ids_all)})
        if not a.smoke:
            pd.DataFrame(timings).to_csv(tim_p, index=False)
        print(f"{cond}: {dt:.0f}s · {nwin / len(ids_all):.1f} ventanas/estudio", flush=True)


if __name__ == "__main__":
    main()
