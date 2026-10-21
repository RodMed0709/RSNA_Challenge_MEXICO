"""Inference primitives for exp091_stress_v1.

The executable GPU runner is added after the deterministic condition contract is
covered by tests. This module never trains or changes source checkpoints.
"""

from __future__ import annotations

import argparse
import math
import json
import sys
import time
import zlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np


PLAUSIBLE_CONDITIONS = (
    "gauss2",
    "gamma0.7",
    "gamma1.4",
    "contr-20",
    "bias",
    "cortes50",
    "shift+2",
    "k12",
)
SEVERE_CONDITIONS = (
    "gauss5",
    "rician5",
    "cortes25",
    "res192",
    "drop_s0",
    "drop_sag",
)
PUBLISHED_CONDITIONS = ("ref", "tta_mirror", *PLAUSIBLE_CONDITIONS, *SEVERE_CONDITIONS)
INFERENCE_CONDITIONS = ("ref", "mirror", *PLAUSIBLE_CONDITIONS, *SEVERE_CONDITIONS)
IMG = 384
BASE_CROP = 320
SLOT_SIZES = (26, 22, 18, 12, 18)
SAGITTAL_SLOTS = (0, 1)


def deterministic_seed(uid: str, condition: str) -> int:
    """Return exp040's deterministic per-study/condition seed."""

    return zlib.crc32(f"{uid}|{condition}".encode()) & 0x7FFFFFFF


def slice_plan(condition: str, slot_sizes: list[int] | tuple[int, ...]):
    """Return volume indices and new slot sizes for a slice perturbation."""

    boundaries = np.cumsum([0, *slot_sizes])
    if condition.startswith("cortes"):
        fraction = {"cortes50": 0.5, "cortes25": 0.25}[condition]
        indices: list[int] = []
        sizes: list[int] = []
        for slot, size in enumerate(slot_sizes):
            count = max(3, int(math.ceil(size * fraction)))
            local = np.linspace(0, size - 1, count).round().astype(int)
            indices.extend((boundaries[slot] + local).tolist())
            sizes.append(count)
        return np.asarray(indices, dtype=np.int64), sizes
    if condition.startswith("shift"):
        displacement = int(condition[5:])
        indices = []
        for slot, size in enumerate(slot_sizes):
            shifted = np.clip(
                boundaries[slot] + np.arange(size) + displacement,
                boundaries[slot],
                boundaries[slot + 1] - 1,
            )
            indices.extend(shifted.tolist())
        return np.asarray(indices, dtype=np.int64), list(slot_sizes)
    return None


def derive_published_predictions(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Derive TTA and remove the unpublished raw mirror condition."""

    if "ref" not in raw or "mirror" not in raw:
        raise ValueError("ref and mirror predictions are required to derive tta_mirror")
    published = {name: value for name, value in raw.items() if name != "mirror"}
    published["tta_mirror"] = (np.asarray(raw["ref"]) + np.asarray(raw["mirror"])) / 2.0
    return published


def load_panel_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_panel_config(config: Mapping, check_paths: bool = False) -> dict:
    panels = config.get("panels")
    if not isinstance(panels, dict) or set(panels) != {"R", "F", "S"}:
        raise ValueError("config panels must be exactly R, F and S")
    models = [model for panel in ("R", "F", "S") for model in panels[panel]]
    ids = [model.get("id") for model in models]
    if any(not model_id for model_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("model ids must be present and globally unique")
    for panel in ("R", "F"):
        for model in panels[panel]:
            if model.get("pseudo_auc") is None or not model.get("oof"):
                raise ValueError(f"panel {panel} requires pseudo_auc and oof for {model['id']}")
    s_ids = {model["id"] for model in panels["S"]}
    ensembles = config.get("derived_ensembles", [])
    for ensemble in ensembles:
        if not set(ensemble.get("members", [])).issubset(s_ids):
            raise ValueError(f"unknown ensemble member in {ensemble.get('id')}")
    if check_paths:
        for model in models:
            for field in ("checkpoint", "oof"):
                path = model.get(field)
                if path and not Path(path).is_file():
                    raise FileNotFoundError(f"{model['id']} {field}: {path}")
    return {
        "panel_counts": {panel: len(panels[panel]) for panel in ("R", "F", "S")},
        "derived_ensembles": len(ensembles),
        "unique_model_ids": len(set(ids)),
    }


def resolve_model_runtime(model: Mapping, checkpoint_meta: Mapping) -> dict:
    params = checkpoint_meta.get("params") or {}
    res = int(model.get("res") or checkpoint_meta.get("res") or params.get("RES_IN") or params.get("RES") or 320)
    slots = model.get("use_slots") or params.get("USE_SLOTS") or [0, 1, 2, 3, 4]
    k_eval = int(model.get("k_eval") or params.get("K_EVAL") or 24)
    if res not in (224, 320):
        raise ValueError(f"unsupported model resolution: {res}")
    slots = tuple(int(slot) for slot in slots)
    if not slots or not set(slots).issubset(range(5)):
        raise ValueError(f"invalid slots: {slots}")
    if k_eval <= 0:
        raise ValueError("k_eval must be positive")
    return {"res": res, "use_slots": slots, "k_eval": k_eval}


def select_cohort_indices(folds: np.ndarray, is_gold: np.ndarray, target_fold: int):
    folds = np.asarray(folds)
    is_gold = np.asarray(is_gold, dtype=bool)
    if folds.shape != is_gold.shape:
        raise ValueError("folds and is_gold must have equal shape")
    validation = np.flatnonzero((folds == target_fold) & ~is_gold)
    gold = np.flatnonzero(is_gold)
    return validation, gold


def writable_array(array: np.ndarray) -> np.ndarray:
    """Return an owned C-contiguous array safe for ``torch.from_numpy``."""

    return np.array(array, copy=True, order="C")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_entry(config: Mapping, panel: str, model_id: str) -> dict:
    matches = [model for model in config["panels"][panel] if model["id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"expected one {panel}/{model_id} model, found {len(matches)}")
    return matches[0]


def _eval_centers(mask: np.ndarray, count: int) -> np.ndarray:
    valid = np.flatnonzero(mask > 0)
    if len(valid) < 3:
        valid = np.arange(min(3, len(mask)))
    lo, hi = int(valid.min()), int(valid.max())
    centers = [center for center in range(lo + 1, hi) if center - 1 >= lo and center + 1 <= hi]
    if not centers:
        centers = [max(1, min((lo + hi) // 2, len(mask) - 2))]
    chosen = np.linspace(0, len(centers) - 1, count).round().astype(int)
    return np.asarray([max(1, min(centers[index], len(mask) - 2)) for index in chosen], dtype=np.int64)


def _condition_k(condition: str, default: int) -> int:
    return 12 if condition == "k12" else default


def _run_gpu(args, config: Mapping, model: Mapping) -> dict:
    import pandas as pd
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    repo = _repo_root()
    sys.path.insert(0, str(repo))
    from src.knee.config import LABELS
    from src.knee.splits import load_split

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stress inference")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    checkpoint = torch.load(model["checkpoint"], map_location="cpu", weights_only=False)
    runtime = resolve_model_runtime(model, checkpoint)

    class RaptorClassifier(nn.Module):
        def __init__(self, backbone, feature_dim=384, labels=12, drop=0.2):
            super().__init__()
            self.backbone = backbone
            self.norm = nn.LayerNorm(feature_dim)
            self.att = nn.Sequential(nn.Linear(feature_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, labels))
            self.clsW = nn.Parameter(torch.zeros(labels, feature_dim))
            self.clsb = nn.Parameter(torch.zeros(labels))

        def head(self, features):
            hidden = self.norm(features)
            attention = torch.softmax(self.att(hidden), dim=1)
            pooled = torch.einsum("bkn,bkf->bnf", attention, hidden)
            return (pooled * self.clsW).sum(-1) + self.clsb

    backbone = timm.create_model(
        checkpoint["arch"],
        pretrained=False,
        num_classes=0,
        in_chans=3,
        global_pool="avg",
        img_size=runtime["res"],
    )
    network = RaptorClassifier(backbone, feature_dim=backbone.num_features)
    network.load_state_dict(checkpoint["model"], strict=True)
    network = network.to(device).eval()
    del checkpoint

    split = load_split("split_v1_seed42")
    split["StudyInstanceUID"] = split.StudyInstanceUID.astype(str)
    val_idx, gold_idx = select_cohort_indices(
        split.fold.to_numpy(), split.is_gold.to_numpy(dtype=bool), int(model.get("fold", 0))
    )
    validation_ids = split.iloc[val_idx].StudyInstanceUID.tolist()
    gold_ids = split.iloc[gold_idx].StudyInstanceUID.tolist()
    ids = validation_ids + gold_ids
    if args.smoke:
        ids = validation_ids[:20]

    corpus = Path(config["corpus"])
    corpus_ids = np.load(corpus / "ids.npy").astype(str)
    masks = np.load(corpus / "masks.npy")
    id_to_row = {uid: index for index, uid in enumerate(corpus_ids)}
    missing = [uid for uid in ids if uid not in id_to_row]
    if missing:
        raise RuntimeError(f"{len(missing)} cohort studies missing from corpus")
    rows = np.asarray([id_to_row[uid] for uid in ids], dtype=np.int64)

    if args.smoke:
        source = np.load(corpus / "vols.npy", mmap_mode="r")
        local = [np.asarray(source[row]) for row in rows]
        get_volume = lambda index: local[index]
    else:
        cache_path = Path(f"/root/exp091_cache_f{int(model.get('fold', 0))}.npy")
        meta_path = cache_path.with_suffix(".json")
        cache_valid = False
        if cache_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cache_valid = meta.get("ids") == ids
        if not cache_valid:
            started = time.time()
            source = np.load(corpus / "vols.npy", mmap_mode="r")
            temporary = cache_path.with_suffix(".tmp.npy")
            destination = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.uint8, shape=(len(rows), sum(SLOT_SIZES), IMG, IMG)
            )
            for ordinal, source_row in enumerate(rows):
                destination[ordinal] = source[source_row]
                if ordinal % 100 == 0:
                    print(f"cache {ordinal}/{len(rows)} {time.time() - started:.0f}s", flush=True)
            destination.flush()
            del destination
            temporary.replace(cache_path)
            meta_path.write_text(json.dumps({"ids": ids}) + "\n", encoding="utf-8")
        cached = np.load(cache_path, mmap_mode="r")
        get_volume = lambda index: cached[index]

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def generator(uid: str, condition: str):
        return torch.Generator(device=device).manual_seed(deterministic_seed(uid, condition))

    def bias_field(random_generator):
        coarse = torch.randn(1, 1, 4, 4, generator=random_generator, device=device)
        field = functional.interpolate(coarse, size=(IMG, IMG), mode="bicubic", align_corners=True)[0, 0]
        field = field / field.abs().max().clamp_min(1e-6)
        return torch.exp(0.2 * field)

    def perturb(volume, valid, condition, random_generator):
        if condition not in {"gauss2", "gauss5", "rician5", "gamma0.7", "gamma1.4", "contr-20", "bias", "res192"}:
            return volume
        values = volume[valid]
        if condition.startswith("gauss"):
            sigma = float(condition[5:]) / 100.0
            values = values + sigma * torch.randn(values.shape, generator=random_generator, device=device)
        elif condition.startswith("rician"):
            sigma = float(condition[6:]) / 100.0
            first = torch.randn(values.shape, generator=random_generator, device=device)
            second = torch.randn(values.shape, generator=random_generator, device=device)
            values = torch.sqrt((values + sigma * first) ** 2 + (sigma * second) ** 2)
        elif condition.startswith("gamma"):
            values = values.clamp_min(0) ** float(condition[5:])
        elif condition.startswith("contr"):
            multiplier = 1.0 + float(condition[5:]) / 100.0
            center = values.mean()
            values = (values - center) * multiplier + center
        elif condition == "bias":
            values = values * bias_field(random_generator)[None]
        elif condition == "res192":
            values = functional.interpolate(values[:, None], size=(192, 192), mode="area")
            values = functional.interpolate(values, size=(IMG, IMG), mode="bilinear", align_corners=False)[:, 0]
        result = volume.clone()
        result[valid] = values.clamp(0, 1)
        return result

    def make_bag(volume_u8, mask, uid: str, condition: str):
        sizes = list(SLOT_SIZES)
        plan = slice_plan(condition, sizes)
        volume = volume_u8
        current_mask = mask.copy()
        if plan is not None:
            indices, sizes = plan
            volume = volume_u8[torch.as_tensor(indices, device=device)]
            current_mask = current_mask[indices]
        boundaries = np.cumsum([0, *sizes])
        for slot in range(len(sizes)):
            if slot not in runtime["use_slots"]:
                current_mask[boundaries[slot]:boundaries[slot + 1]] = 0
        drops = {"drop_s0": (0,), "drop_sag": (0, 1)}.get(condition, ())
        for slot in drops:
            current_mask[boundaries[slot]:boundaries[slot + 1]] = 0
        centers = _eval_centers(current_mask, _condition_k(condition, runtime["k_eval"]))
        needed = np.unique(np.concatenate([centers - 1, centers, centers + 1]))
        remap = np.full(len(current_mask), -1, dtype=np.int64)
        remap[needed] = np.arange(len(needed))
        values = volume[torch.as_tensor(needed, device=device)].float() / 255.0
        valid_needed = torch.as_tensor(current_mask[needed] > 0, device=device)
        values = values * valid_needed.view(-1, 1, 1)
        values = perturb(values, valid_needed, condition, generator(uid, condition))
        triple_indices = np.stack([remap[centers - 1], remap[centers], remap[centers + 1]], axis=1)
        triples = values[torch.as_tensor(triple_indices, device=device)]
        if condition == "mirror":
            slot_of = np.repeat(np.arange(len(sizes)), sizes)
            sagittal = torch.as_tensor(np.isin(slot_of[centers], SAGITTAL_SLOTS), device=device).view(-1, 1, 1, 1)
            triples = torch.where(sagittal, triples.flip(1), triples.flip(-1))
        offset = (IMG - BASE_CROP) // 2
        triples = triples[:, :, offset:offset + BASE_CROP, offset:offset + BASE_CROP]
        if runtime["res"] != BASE_CROP:
            triples = functional.interpolate(
                triples, size=(runtime["res"], runtime["res"]), mode="bilinear", align_corners=False
            )
        return (triples - mean) / std

    @torch.no_grad()
    def forward_bags(bags, chunk=96):
        flat = torch.cat(bags, dim=0)
        features = []
        for start in range(0, len(flat), chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features.append(network.backbone(flat[start:start + chunk]).float())
        features = torch.cat(features, dim=0)
        output, offset = [], 0
        for bag in bags:
            current = features[offset:offset + len(bag)][None]
            offset += len(bag)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output.append(torch.sigmoid(network.head(current).float())[0])
        return torch.stack(output).cpu().numpy()

    requested = tuple(args.conditions.split(",")) if args.conditions else INFERENCE_CONDITIONS
    if args.smoke and not args.conditions:
        requested = ("ref", "mirror", "gauss2")
    unknown = set(requested) - set(INFERENCE_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    output = Path(args.output) if args.output else repo / "results" / "exp091_stress_v1" / "private" / args.panel / model["id"]
    output.mkdir(parents=True, exist_ok=True)
    timings_path = output / "timings.json"
    timings = json.loads(timings_path.read_text(encoding="utf-8")) if timings_path.exists() else {}
    for condition in requested:
        destination = output / f"{condition}.parquet"
        if destination.exists() and not args.force:
            print(f"skip {args.panel}/{model['id']}/{condition}", flush=True)
            continue
        started = time.time()
        predictions = np.zeros((len(ids), len(LABELS)), dtype=np.float32)
        window_count = 0
        for start in range(0, len(ids), args.bs):
            bags = []
            for index in range(start, min(start + args.bs, len(ids))):
                volume = torch.from_numpy(writable_array(get_volume(index))).to(device, non_blocking=True)
                bags.append(make_bag(volume, masks[rows[index]], ids[index], condition))
            window_count += sum(len(bag) for bag in bags)
            predictions[start:start + len(bags)] = forward_bags(bags)
        torch.cuda.synchronize()
        frame = pd.DataFrame(predictions, index=pd.Index(ids, name="StudyInstanceUID"), columns=LABELS)
        temporary_predictions = destination.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary_predictions)
        temporary_predictions.replace(destination)
        timings[condition] = {
            "seconds": time.time() - started,
            "n": len(ids),
            "windows_mean": window_count / len(ids),
        }
        temporary = timings_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
        temporary.replace(timings_path)
        print(
            f"{args.panel}/{model['id']}/{condition}: {timings[condition]['seconds']:.1f}s "
            f"n={len(ids)} windows={timings[condition]['windows_mean']:.1f}",
            flush=True,
        )
    return {"panel": args.panel, "model": model["id"], "n": len(ids), "conditions": list(requested), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="exp091 stress inference (no training)")
    parser.add_argument("--config", default=str(Path(__file__).with_name("panel_config.json")))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--panel", choices=("R", "F", "S"))
    parser.add_argument("--model")
    parser.add_argument("--conditions", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config = load_panel_config(args.config)
    summary = validate_panel_config(config, check_paths=True)
    if args.validate_only:
        print(json.dumps(summary, sort_keys=True))
        return 0
    if not args.panel or not args.model:
        parser.error("--panel and --model are required unless --validate-only is used")
    model = _model_entry(config, args.panel, args.model)
    result = _run_gpu(args, config, model)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
