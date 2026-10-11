#!/usr/bin/env python3
"""exp041_t4_sin_slots -- build the Kaggle kernels (nothing of the third-party notebooks goes to git).

  python make_kernel.py --nb <rsna-knee-0949-anatomical-mirror.ipynb> --nb224 <rsna-knee-0945-efficient-224crop.ipynb>
                        --studies exp041_studies.csv --out <dir> [--submit V ...]

Writes
  <out>/bench/        private kernel medellinrodrigo/exp041-t4-sin-slots: injects both notebooks' code cells (base64)
                      + t4_slots.py + the study list (no labels) and runs the timing harness on train studies.
  <out>/submit_<V>/   submission kernels (only with --submit): the public notebook of the variant's model with its
                      code cell patched (see SUBMIT below). Every patch anchor must hit exactly once.
Same sources / accelerator as the public notebooks; docker image not pinned (as exp026). Nothing is pushed here.
"""
import argparse, base64, copy, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DS = "nartaa/rsna-knee-publication-swa-weights-20261007"
COMP = "rsna-knee-abnormality-detection"


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def metadata(slug, title, code_file, docker):
    m = {"id": f"medellinrodrigo/{slug}", "title": title, "code_file": code_file, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "keywords": ["gpu"], "dataset_sources": [WEIGHTS_DS],
            "kernel_sources": [], "competition_sources": [COMP], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4"}
    if docker:
        m["docker_image"] = docker
    return m


def b64(s):
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def write_kernel(d, nb, meta):
    os.makedirs(d, exist_ok=True)
    json.dump(nb, open(os.path.join(d, meta["code_file"]), "w", encoding="utf-8"), indent=1)
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)


# ---- submission patches (each must hit exactly once) ------------------------------------------
BENCH_OFF = [("torch.backends.cudnn.benchmark = True\n",
              "torch.backends.cudnn.benchmark = False   # exp027: no per-thread cudnn autotune\n")]
# 0949 notebook (384 model): R = exp026 V2 (channels_last + no TTA)
P_CL = [
    ("    model.eval().to(device)\n",
     "    model.eval().to(device)\n    model.to(memory_format=torch.channels_last)   # exp026 V1\n"),
    ("f = torch.cat([self.backbone(chunk) for chunk in flat.split(16)], dim=0)",
     "f = torch.cat([self.backbone(chunk.contiguous(memory_format=torch.channels_last)) "
     "for chunk in flat.split(16)], dim=0)   # exp026 V1"),
]
TTA_BODY = ("                    xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)\n"
            "                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)\n"
            "                    del xm\n")
P_NOTTA = [(TTA_BODY, "                    arm_probs[a][i] = infer_probs(model, xw, d)   # exp026 V2: no TTA\n")]
P384_HALF = [
    ("    model.to(memory_format=torch.channels_last)   # exp026 V1\n",
     "    model.half().to(memory_format=torch.channels_last)   # exp026 V1 + exp027 C (model.half())\n"),
    ('    x = xwins.unsqueeze(0).to(device)\n    with torch.autocast("cuda", dtype=torch.float16):\n',
     "    x = xwins.unsqueeze(0).to(device).half()   # exp027 C: pure fp16, no autocast\n    if True:\n"),
]
# 0945 notebook (224 model): as published = B94 with cudnn.benchmark and no channels_last
P224_CL = [("    model.eval().to(device)\n",
            "    model.eval().to(device)\n    model.to(memory_format=torch.channels_last)   # exp027\n")]
P224_CHUNK16 = [("        f = self.backbone(x.flatten(0, 1))\n",
                 "        f = torch.cat([self.backbone(c.contiguous(memory_format=torch.channels_last))\n"
                 "                       for c in x.flatten(0, 1).split(16)], dim=0)   # exp027: chunks of 16, channels_last\n")]
P224_ONEBATCH_CL = [("        f = self.backbone(x.flatten(0, 1))\n",
                     "        f = self.backbone(x.flatten(0, 1).contiguous(memory_format=torch.channels_last))   # exp027\n")]
P224_HALF = [
    ("    model.to(memory_format=torch.channels_last)   # exp027\n",
     "    model.half().to(memory_format=torch.channels_last)   # exp027: model.half()\n"),
    ("    x = xwins.unsqueeze(0).to(device)\n", "    x = xwins.unsqueeze(0).to(device).half()   # exp027: fp16 input\n"),
    ('            with torch.autocast("cuda", dtype=torch.float16):\n                o = torch.sigmoid(model(x).float())\n',
     "            if True:   # exp027: pure fp16 (model.half()), no autocast\n                o = torch.sigmoid(model(x).float())\n"),
]
SUBMIT = {   # variant -> (notebook key, patches, description)
    "R":   ("384", P_CL + P_NOTTA, "exp026 V2: channels_last + no mirror TTA"),
    "A":   ("384", BENCH_OFF + P_CL + P_NOTTA, "channels_last + no mirror TTA + cudnn.benchmark off"),
    "C":   ("384", BENCH_OFF + P_CL + P_NOTTA + P384_HALF, "channels_last + no TTA + cudnn.benchmark off + model.half()"),
    "B":   ("224", BENCH_OFF + P224_CL + P224_CHUNK16, "224 model, channels_last, chunks of 16, cudnn.benchmark off"),
    "B94": ("224", BENCH_OFF + P224_CL + P224_ONEBATCH_CL, "224 model, channels_last, one batch of 94, cudnn.benchmark off"),
    "E1":  ("224", BENCH_OFF + P224_CL + P224_CHUNK16 + P224_HALF,
            "224 model, channels_last, chunks of 16, cudnn.benchmark off, model.half()"),
    "E1_94": ("224", BENCH_OFF + P224_CL + P224_ONEBATCH_CL + P224_HALF,
              "224 model, channels_last, one batch of 94, cudnn.benchmark off, model.half()"),
}
# exp041: drop slots 3 (Cor non-fluid) and 4 (Axial): 3-slot read (66 slices) and 64 windows
P_SLOTS3 = [
    ("    _READER = _load_fastread().make_builder(FASTREAD_VARIANT, IMG, SLOTS, hdr_threads=FR_HDR, px_threads=FR_PX)\n",
     "    _READER = _load_fastread().make_builder(FASTREAD_VARIANT, IMG, SLOTS[:3], hdr_threads=FR_HDR, px_threads=FR_PX)"
     "   # exp041: no Cor non-fluid / Axial I/O\n"),
    ("K_EVAL = 94   #", "K_EVAL = 64   # exp041: 64 window centres of the 66-slice 3-slot volume;"),
]
# C3t: mirror TTA needs the 96-slice layout (_slot_index): zero-pad the 66-slice volume in the consumer
P_TTA96 = [
    ("            sid, vol, mask = item; i = pos[sid]\n",
     "            sid, vol, mask = item; i = pos[sid]\n"
     "            if vol.shape[0] < MAXS:   # exp041: pad the 3-slot volume to the 96-slice layout (zeros, mask 0)\n"
     "                vol = np.concatenate([vol, np.zeros((MAXS - vol.shape[0],) + vol.shape[1:], vol.dtype)])\n"
     "                mask = np.concatenate([mask, np.zeros(MAXS - mask.shape[0], mask.dtype)])\n"),
    ("    assert plain.shape==mirrored.shape==(94,3,320,320)\n",
     "    assert plain.shape==mirrored.shape==(K_EVAL,3,320,320)   # exp041\n"),
]
SUBMIT.update({
    "C3":  ("384", BENCH_OFF + P_CL + P_NOTTA + P384_HALF + P_SLOTS3,
            "no mirror TTA + model.half() + channels_last + cudnn.benchmark off + no Cor non-fluid / Axial slots (64 windows)"),
    "C3t": ("384", BENCH_OFF + P_CL + P384_HALF + P_SLOTS3 + P_TTA96,
            "mirror TTA kept + model.half() + channels_last + cudnn.benchmark off + no Cor non-fluid / Axial slots (64 windows)"),
    "C3h": ("224", BENCH_OFF + P224_CL + P224_ONEBATCH_CL + P224_HALF + P_SLOTS3,
            "224 model, one batch, model.half(), channels_last, cudnn.benchmark off, no Cor non-fluid / Axial slots (64 windows)"),
})
SRC_NB = {"384": "nartaa/rsna-knee-0949-anatomical-mirror", "224": "nartaa/rsna-knee-0945-efficient-224crop"}


def patch(src, pairs):
    for a, b in pairs:
        n = src.count(a)
        assert n == 1, f"patch anchor found {n} times: {a[:60]!r}"
        src = src.replace(a, b)
    return src


def main_cell(nb):
    ci = max((i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"),
             key=lambda i: len("".join(nb["cells"][i]["source"])))
    return ci, "".join(nb["cells"][ci]["source"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", required=True)
    ap.add_argument("--nb224", required=True)
    ap.add_argument("--studies", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--submit", nargs="*", default=[])
    a = ap.parse_args()
    nbs = {"384": json.load(open(a.nb, encoding="utf-8")), "224": json.load(open(a.nb224, encoding="utf-8"))}
    cells = {k: main_cell(v) for k, v in nbs.items()}

    # ---- bench kernel ----
    st = pd.read_csv(a.studies)
    sids = {r: st.loc[st.role == r, "StudyInstanceUID"].astype(str).tolist() for r in ("main", "warmup", "cold5", "cold3")}
    harness = open(os.path.join(HERE, "t4_slots.py"), encoding="utf-8").read()
    bench = {"cells": [
        md_cell("# exp041_t4_sin_slots\n\nTiming of the public Raptor inference without the Coronal non-fluid and Axial slots "
                "(`nartaa/rsna-knee-0949-anatomical-mirror` and `nartaa/rsna-knee-0945-efficient-224crop`, code cells "
                "injected verbatim) on Kaggle T4 x2, TRAIN studies of the competition input. No submission.\n"),
        code_cell("import base64, json\n"
                  f"open('/kaggle/working/raptor_cell.py', 'wb').write(base64.b64decode('{b64(cells['384'][1])}'))\n"
                  f"open('/kaggle/working/cell224.py', 'wb').write(base64.b64decode('{b64(cells['224'][1])}'))\n"
                  f"json.dump(json.loads(base64.b64decode('{b64(json.dumps(sids))}')), "
                  "open('/kaggle/working/exp041_sids.json', 'w'))\n"
                  f"open('/kaggle/working/t4_slots.py', 'wb').write(base64.b64decode('{b64(harness)}'))\n"
                  "print('written')\n"),
        code_cell("import subprocess, sys\n"
                  "p = subprocess.Popen([sys.executable, '-u', '/kaggle/working/t4_slots.py'], cwd='/kaggle/working',\n"
                  "                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
                  "for line in p.stdout: print(line, end='', flush=True)\n"
                  "assert p.wait() == 0, f'harness exit {p.returncode}'\n"),
        code_cell("import os\nfor f in ('_fastread.py', 'raptor_cell.py', 'cell224.py', 'bb224_half.onnx'):\n"
                  "    if os.path.exists('/kaggle/working/' + f): os.remove('/kaggle/working/' + f)\n"
                  "print(sorted(os.listdir('/kaggle/working')))\n"),
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    write_kernel(os.path.join(a.out, "bench"), bench,
                 metadata("exp041-t4-sin-slots", "exp041-t4-sin-slots", "exp041-t4-sin-slots.ipynb", None))

    # ---- submission kernels ----
    for v in a.submit:
        key, pairs, desc = SUBMIT[v]
        nbv = copy.deepcopy(nbs[key]); ci, cell = cells[key]
        nbv["cells"][ci]["source"] = patch(cell, pairs).splitlines(keepends=True)
        for c in nbv["cells"]:
            if c["cell_type"] == "code":
                c["outputs"] = []; c["execution_count"] = None
        nbv["cells"].insert(0, md_cell(f"# exp041 {v}\n\nPrivate copy of `{SRC_NB[key]}` (credits below) with: {desc}.\n"))
        slug = f"exp041-sub-{v.lower().replace('_', '-')}"
        write_kernel(os.path.join(a.out, f"submit_{v}"), nbv, metadata(slug, slug, f"{slug}.ipynb", None))
    print("kernels written to", a.out)


if __name__ == "__main__":
    main()
