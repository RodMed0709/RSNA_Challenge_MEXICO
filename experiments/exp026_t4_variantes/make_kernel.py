#!/usr/bin/env python3
"""exp026_t4_variantes -- build the Kaggle kernels (nothing of the third-party notebook goes to git).

  python make_kernel.py --nb <rsna-knee-0949-anatomical-mirror.ipynb> --studies exp026_studies.csv --out <dir>

Writes
  <out>/bench/       private kernel medellinrodrigo/exp026-t4-variantes: injects the notebook's code cell
                     (base64) + t4_variants.py + the study list (no labels) and runs the timing harness
                     on train studies of the competition input.
  <out>/submit_V1..V4/ submission kernels: the original notebook with its code cell patched for the
                     variant (V1 channels_last; V2 V1 no TTA; V3 V1 sagittal-only TTA; V4 V1 47 windows).
Same sources / docker image / accelerator as the public notebook. Nothing is pushed here.
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
SAG_FN = '''

@torch.no_grad()
def infer_sag_tta(model, vol, mask, xw, device):
    """exp026 V3: mirror TTA on the sagittal windows only. Plain pass on all K_EVAL windows; the mirrored
    bag reuses the plain features of coronal/axial windows and recomputes only the mirrored sagittal ones."""
    D = vol.shape[0]
    cs = np.array([max(1, min(c, D - 2)) for c in _eval_centers(mask, D, K_EVAL)])
    s_, _ = _slot_index(D)
    sagi = torch.from_numpy(np.flatnonzero(np.isin(s_[cs], SAGITTAL_SLOTS))).to(device)
    xm = gpu_windows(vol, mask, K_EVAL, INPUT_RES, device, norm=NORM, mirror=True)[sagi]
    with torch.autocast("cuda", dtype=torch.float16):
        f = model.encode(xw.unsqueeze(0).to(device))
        fm = f.clone()
        if len(sagi):
            fm[0, sagi] = model.encode(xm.unsqueeze(0))[0].to(fm.dtype)
        lp = model.head(f).float(); lq = model.head(fm).float()
        assert torch.isfinite(lp).all() and torch.isfinite(lq).all(), "Nonfinite fp16 logits"
        p = 0.5 * (torch.sigmoid(lp) + torch.sigmoid(lq))
    return p[0].cpu().numpy()

'''
P_SAG = [(TTA_BODY, "                    arm_probs[a][i] = infer_sag_tta(model, vol, mask, xw, d)   # exp026 V3\n"),
         ("\ndef rankpct(x):", SAG_FN + "\ndef rankpct(x):")]
P_K47 = [("K_EVAL = 94   #", "K_EVAL = 47   # exp026 V4 (every 2nd window);"),
         ("    assert plain.shape==mirrored.shape==(94,3,320,320)\n",
          "    assert plain.shape==mirrored.shape==(K_EVAL,3,320,320)   # exp026 V4\n")]
VARIANTS = {"V1": P_CL, "V2": P_CL + P_NOTTA, "V3": P_CL + P_SAG, "V4": P_CL + P_K47}
DESC = {"V1": "channels_last (model + window chunks); otherwise identical (94 windows, mirror TTA)",
        "V2": "channels_last + no mirror TTA (one pass of 94 windows)",
        "V3": "channels_last + mirror TTA on the sagittal windows only (~141 window forwards instead of 188)",
        "V4": "channels_last + 47 windows (every 2nd) with mirror TTA"}


def patch(src, pairs):
    for a, b in pairs:
        n = src.count(a)
        assert n == 1, f"patch anchor found {n} times: {a[:60]!r}"
        src = src.replace(a, b)
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", required=True)
    ap.add_argument("--studies", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--docker", default=None, help="default: docker_image of the notebook's kernel-metadata.json")
    ap.add_argument("--no-docker", action="store_true", help="do not pin the image (latest Kaggle environment)")
    a = ap.parse_args()
    nb = json.load(open(a.nb, encoding="utf-8"))
    docker = a.docker
    if a.no_docker:
        docker = None
    elif docker is None:
        meta_p = os.path.join(os.path.dirname(a.nb), "kernel-metadata.json")
        docker = json.load(open(meta_p))["docker_image"]
    ci = max((i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"),
             key=lambda i: len("".join(nb["cells"][i]["source"])))
    cell = "".join(nb["cells"][ci]["source"])

    # ---- bench kernel ----
    st = pd.read_csv(a.studies)
    sids = {r: st.loc[st.role == r, "StudyInstanceUID"].astype(str).tolist() for r in ("main", "serial", "warmup")}
    harness = open(os.path.join(HERE, "t4_variants.py"), encoding="utf-8").read()
    bench = {"cells": [
        md_cell("# exp026_t4_variantes\n\nTiming of the public Raptor/0.949 inference "
                "(`nartaa/rsna-knee-0949-anatomical-mirror`, code cell injected verbatim) on Kaggle T4 x2, "
                "5 forward variants, 150 TRAIN studies of the competition input. No submission.\n"),
        code_cell("import base64, json\n"
                  f"open('/kaggle/working/raptor_cell.py', 'wb').write(base64.b64decode('{b64(cell)}'))\n"
                  f"json.dump(json.loads(base64.b64decode('{b64(json.dumps(sids))}')), "
                  "open('/kaggle/working/exp026_sids.json', 'w'))\n"
                  f"open('/kaggle/working/t4_variants.py', 'wb').write(base64.b64decode('{b64(harness)}'))\n"
                  "print('written')\n"),
        code_cell("import subprocess, sys\n"
                  "p = subprocess.Popen([sys.executable, '-u', '/kaggle/working/t4_variants.py'], cwd='/kaggle/working',\n"
                  "                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
                  "for line in p.stdout: print(line, end='', flush=True)\n"
                  "assert p.wait() == 0, f'harness exit {p.returncode}'\n"),
        code_cell("import os\nfor f in ('_fastread.py', 'raptor_cell.py'):\n"
                  "    if os.path.exists('/kaggle/working/' + f): os.remove('/kaggle/working/' + f)\n"
                  "print(sorted(os.listdir('/kaggle/working')))\n"),
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    write_kernel(os.path.join(a.out, "bench"), bench,
                 metadata("exp026-t4-variantes", "exp026-t4-variantes", "exp026-t4-variantes.ipynb", docker))

    # ---- submission kernels ----
    for v, pairs in VARIANTS.items():
        nbv = copy.deepcopy(nb)
        nbv["cells"][ci]["source"] = patch(cell, pairs).splitlines(keepends=True)
        for c in nbv["cells"]:
            if c["cell_type"] == "code":
                c["outputs"] = []; c["execution_count"] = None
        nbv["cells"].insert(0, md_cell(f"# exp026 {v}\n\nPrivate copy of `nartaa/rsna-knee-0949-anatomical-mirror` "
                                       f"(credits below) with one change: {DESC[v]}.\n"))
        slug = f"exp026-sub-{v.lower()}"
        write_kernel(os.path.join(a.out, f"submit_{v}"), nbv, metadata(slug, slug, f"{slug}.ipynb", docker))
    print("kernels written to", a.out)


if __name__ == "__main__":
    main()
