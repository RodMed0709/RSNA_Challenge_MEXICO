"""exp066: build the compact px cache (px_cache.py format, views v4, 384 px) for the training studies from raw DICOM.

Path per study = heliosli's inference path (compact_infer.CStudies): training.native_cache.prepare -> geometry bank ->
side-classifier check (re-render when confidently mirrored) -> SKM-TEA knee anchors -> px_cache.StudySlices.build.
That is the same code the public bundle uses at test time; we write its rows into the px_cache layout
(px.u8.npy [N,S,S], meta.f32.npy [N,12], index.json) that compact_train.Studies reads.
No labels are read here.  Heliosli code is imported from --src/--code (public_assets copy, SHA-checked), never copied
into git.
usage: python build_px_cache.py --uids uids.txt --out /root/px384_v4 --workers 20 [--limit N]
"""
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch

META_COLS = ["series_idx", "slot", "plane", "view_id", "rel_pos", "log_spacing", "extent_130", "fluid",
             "fatsat", "slice_order", "n_slices", "real_series_idx"]


class Builder(torch.utils.data.Dataset):
    def __init__(self, a, uids):
        self.a, self.uids, self.cs = a, uids, None

    def __len__(self):
        return len(self.uids)

    def _setup(self):
        for p in (self.a.code, self.a.src):
            if p not in sys.path:
                sys.path.insert(0, p)
        import compact_infer as CI
        wid = torch.utils.data.get_worker_info()
        tmp = Path(self.a.tmp) / ("w%d" % (wid.id if wid else 0))
        tmp.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(1)
        self.cs = CI.CStudies(self.a.data, "train", [], tmp, self.a.code, self.a.size, eval_stride=1, views="v4", slices=3,
                              anchor_weights=self.a.anchor_weights, phys_mm=3.5, side_model=self.a.side_model,
                              side_threshold=2.0, side_max_x=0.0, side_position_mm=20.0, extent_meta=True)
        self.cs._setup()

    def __getitem__(self, i):
        uid = self.uids[i]
        t0 = time.time()
        try:
            if self.cs is None:
                self._setup()
            cs = self.cs
            work = Path(cs.tmp) / ("s_" + hashlib.sha256(uid.encode()).hexdigest()[:16])
            try:
                side = cs._native_and_geometry(uid, work)
                cs.builder.cache = str(work)
                key = cs.KB.study_key(uid)
                with open(work / "studies" / (key + ".json"), encoding="utf-8") as fh:
                    study = json.load(fh)
                rec = cs.bank.record(uid, study, geo_dir=str(work / "geometry"))
                (work / "anchors").mkdir(exist_ok=True)
                with open(work / "anchors" / (key + ".json"), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh)
                cs.builder.anchors = str(work / "anchors")
                item = cs.builder.build(uid)
            finally:
                shutil.rmtree(work, ignore_errors=True)
            item["side"] = side
            item["anchor"] = {"method": rec.get("method")}
        except Exception as exc:
            item = {"uid": uid, "error": repr(exc)[:400], "px": torch.zeros((0, self.a.size, self.a.size), dtype=torch.uint8),
                    "meta": torch.zeros((0, len(META_COLS))), "blocks": [], "removed": 0}
        item["seconds"] = round(time.time() - t0, 2)
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/workspace/data/raw")
    ap.add_argument("--src", default="/root/compact/src")
    ap.add_argument("--code", default="/root/compact/code")
    ap.add_argument("--anchor-weights", default="/root/compact/anatomy.pt")
    ap.add_argument("--side-model", default="/root/compact/side_cls.pt")
    ap.add_argument("--tmp", default="/root/ctmp")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--rows-per-study", type=int, default=1100, help="memmap upper bound per study (trimmed at the end)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    uids = [u.strip() for u in open(a.uids) if u.strip()]
    if a.limit:
        uids = uids[:a.limit]
    S = a.size
    index_path = os.path.join(a.out, "index.json")
    if os.path.exists(index_path):
        index = json.load(open(index_path))
    else:
        index = {"order": uids, "done": {}, "blocks": {}, "cursor": 0, "errors": {}, "removed_series": 0, "meta_cols": META_COLS,
                 "side": {}, "seconds": {},
                 "config": {"size": S, "views": "v4", "policy": "series_companded", "builder": "exp066 build_px_cache.py "
                            "(compact_infer.CStudies path: native -> geometry -> side check -> anchors -> StudySlices.build)"}}
    N = len(uids) * a.rows_per_study
    pxp, mp = os.path.join(a.out, "px.u8.npy"), os.path.join(a.out, "meta.f32.npy")
    if os.path.exists(pxp):
        px_mm, meta_mm = np.load(pxp, mmap_mode="r+"), np.load(mp, mmap_mode="r+")
    else:
        px_mm = np.lib.format.open_memmap(pxp, mode="w+", dtype=np.uint8, shape=(N, S, S))
        meta_mm = np.lib.format.open_memmap(mp, mode="w+", dtype=np.float32, shape=(N, len(META_COLS)))
    todo = [u for u in uids if u not in index["done"] and u not in index["errors"]]
    dl = torch.utils.data.DataLoader(Builder(a, todo), batch_size=None, shuffle=False, num_workers=a.workers,
                                     prefetch_factor=2 if a.workers else None, persistent_workers=False)
    t0, last, k = time.time(), time.time(), 0

    def save(final=False):
        px_mm.flush(); meta_mm.flush()
        json.dump(index, open(index_path + ".tmp", "w")); os.replace(index_path + ".tmp", index_path)
        el = time.time() - t0
        print(json.dumps({"done": len(index["done"]), "errors": len(index["errors"]), "todo": len(todo), "rows": index["cursor"],
                          "elapsed_min": round(el / 60, 1), "eta_min": round((len(todo) - k) * el / max(k, 1) / 60, 1),
                          "final": final}), flush=True)

    for item in dl:
        k += 1
        uid = item["uid"]
        if item["error"]:
            index["errors"][uid] = item["error"]
            print("ERROR", uid, item["error"][:200], flush=True)
            continue
        n, cur = len(item["px"]), index["cursor"]
        if cur + n > N:
            raise RuntimeError("row bound exceeded; raise --rows-per-study")
        if n:
            px_mm[cur:cur + n] = item["px"].numpy()
            meta_mm[cur:cur + n] = item["meta"].numpy()
        for b in item["blocks"]:
            b["views"] = [[v, cur + off] for v, off in b["views"]]
        index["done"][uid] = [cur, n, len(item["blocks"])]
        index["blocks"][uid] = item["blocks"]
        index["cursor"] = cur + n
        index["removed_series"] += int(item.get("removed", 0))
        index["side"][uid] = item.get("side")
        index["seconds"][uid] = item["seconds"]
        if time.time() - last > 120:
            save(); last = time.time()
    save(final=True)
    print("DONE rows", index["cursor"], "studies", len(index["done"]), "errors", len(index["errors"]),
          "minutes %.1f" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
