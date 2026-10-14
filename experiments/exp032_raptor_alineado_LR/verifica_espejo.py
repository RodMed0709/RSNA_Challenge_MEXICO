"""exp032 - verificación del espejo L/R sobre el corpus Raptor (script auxiliar, no entrena).

Plantilla por slot = media de N_TEMPLATE rodillas derechas (nativas = alineadas). Para 20 izquierdas y 20 derechas de
control (fuera de la plantilla) mide la correlación de Pearson del bloque del slot (recorte central 320 → 40×40 por
corte, z-score) con la plantilla, nativo vs alineado. Si el espejo es correcto: izquierdas alineado > nativo.
Guarda una figura con el corte central coronal/axial y un sagital lateral, nativo | alineado | plantilla.
Uso: python verifica_espejo.py --corpus /root/corpus96_384 --codes /root/corpus96_384/align_lr_exp032 --png out.png
"""
import argparse, json

import numpy as np
import pandas as pd

SLOTS = [("Sag fluid", 26), ("Sag", 22), ("Cor fluid", 18), ("Cor", 12), ("Ax", 18)]
OFF = np.cumsum([0] + [k for _, k in SLOTS])
IMG, RES = 384, 320


def align(vol, mask, codes):
    """Aplica el espejo por slot (mismo código que el notebook). vol (96,H,W), mask (96,)."""
    vol, mask = vol.copy(), mask.copy()
    for s, c in enumerate(codes):
        a, b = OFF[s], OFF[s + 1]
        if c == 1:
            vol[a:b], mask[a:b] = vol[a:b][::-1], mask[a:b][::-1]
        elif c == 2:
            vol[a:b] = vol[a:b, ::-1]
        elif c == 3:
            vol[a:b] = vol[a:b, :, ::-1]
    return vol, mask


def feat(vol, s):
    a, b = OFF[s], OFF[s + 1]
    lo = (IMG - RES) // 2
    x = vol[a:b, lo:lo + RES, lo:lo + RES].astype(np.float32)
    x = x.reshape(b - a, 40, 8, 40, 8).mean((2, 4))
    return (x - x.mean()) / (x.std() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/root/corpus96_384")
    ap.add_argument("--codes", default="/root/corpus96_384/align_lr_exp032")
    ap.add_argument("--n_template", type=int, default=300)
    ap.add_argument("--png", default="/root/exp032/verifica_espejo.png")
    a = ap.parse_args()
    vols = np.load(f"{a.corpus}/vols.npy", mmap_mode="r")
    masks = np.load(f"{a.corpus}/masks.npy")
    codes = np.load(a.codes + ".npy")
    tab = pd.read_csv(a.codes + ".csv")
    side = tab.drop_duplicates("row").set_index("row").side
    rng = np.random.default_rng(0)
    full = [r for r in range(len(codes)) if all(masks[r, OFF[s]:OFF[s + 1]].sum() == SLOTS[s][1] for s in (0, 2, 4))]
    R = [r for r in full if side.get(r) == "R"]; L = [r for r in full if side.get(r) == "L"]
    rng.shuffle(R); rng.shuffle(L)
    tmpl_rows, ctrl_R, test_L = R[:a.n_template], R[a.n_template:a.n_template + 20], L[:20]
    unk = [r for r in range(len(codes)) if side.get(r, "unknown") == "unknown"]
    res = {"unknown_studies": len(unk), "unknown_with_any_flip": int((codes[unk] != 0).any(1).sum()),
           "left_rows": [int(r) for r in test_L], "per_slot": {}}
    for s in (0, 2, 4):
        T = np.mean([feat(vols[r], s) for r in tmpl_rows], 0)
        T = (T - T.mean()) / T.std()
        def corr(x): return float((x * T).mean())
        nat_L, ali_L, nat_R = [], [], []
        for r in test_L:
            v = np.asarray(vols[r]); nat_L.append(corr(feat(v, s))); ali_L.append(corr(feat(align(v, masks[r], codes[r])[0], s)))
        for r in ctrl_R:
            nat_R.append(corr(feat(np.asarray(vols[r]), s)))
        nat_L, ali_L = np.array(nat_L), np.array(ali_L)
        res["per_slot"][SLOTS[s][0]] = {
            "L_native_mean": float(nat_L.mean()), "L_aligned_mean": float(ali_L.mean()),
            "L_aligned_gt_native": int((ali_L > nat_L).sum()), "n_L": len(test_L),
            "R_control_mean": float(np.mean(nat_R)), "n_R": len(ctrl_R)}
    print(json.dumps(res, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        show = test_L[:6]
        picks = [("Cor fluid", OFF[2] + 9), ("Ax", OFF[4] + 9), ("Sag fluid", OFF[0] + 4)]
        fig, axs = plt.subplots(len(show) + 1, 6, figsize=(15, 2.6 * (len(show) + 1)))
        for i, r in enumerate(show):
            v = np.asarray(vols[r]); va, _ = align(v, masks[r], codes[r])
            for j, (nm, z) in enumerate(picks):
                axs[i, 2 * j].imshow(v[z], cmap="gray"); axs[i, 2 * j].set_title(f"L r{r} {nm} nativo", fontsize=8)
                axs[i, 2 * j + 1].imshow(va[z], cmap="gray"); axs[i, 2 * j + 1].set_title(f"{nm} alineado", fontsize=8)
        for j, (nm, z) in enumerate(picks):
            m = np.mean([np.asarray(vols[r][z], np.float32) for r in tmpl_rows[:100]], 0)
            axs[-1, 2 * j].imshow(m, cmap="gray"); axs[-1, 2 * j].set_title(f"media 100 R {nm}", fontsize=8)
            axs[-1, 2 * j + 1].axis("off")
        for ax in axs.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(); plt.savefig(a.png, dpi=60)
        print("png", a.png)
    except Exception as e:
        print("sin figura:", e)


if __name__ == "__main__":
    main()
