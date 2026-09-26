import time

import torch
from torch.utils.flop_counter import FlopCounterMode


def dummy_study(max_series, n_slices, image_size, device):
    images = torch.zeros(1, max_series, n_slices, 3, image_size, image_size, device=device)
    meta = torch.zeros(1, max_series, 3, dtype=torch.long, device=device)
    mask = torch.ones(1, max_series, dtype=torch.bool, device=device)
    return images, meta, mask


@torch.no_grad()
def measure_cost(model, max_series, n_slices, image_size, device, batch=4, repeats=10):
    model.eval()
    inputs = dummy_study(max_series, n_slices, image_size, device)
    with FlopCounterMode(display=False) as counter:
        model(*inputs)
    gmacs = counter.get_total_flops() / 2 / 1e9

    images, meta, mask = (x.expand(batch, *x.shape[1:]).contiguous() for x in inputs)
    with model.autocast_context(device):
        for _ in range(2):
            model(images, meta, mask)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(images, meta, mask)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / (repeats * batch)
    return {
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "gmacs_per_study": gmacs,
        "ms_per_study": seconds * 1000,
        "device": torch.cuda.get_device_name() if torch.device(device).type == "cuda" else "cpu",
    }
