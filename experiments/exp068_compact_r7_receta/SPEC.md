# exp068_compact_r7_receta

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp066_compact_f0 (compact sola fold 0: pseudo L2 0.8755, gold 0.8972) |
| Fecha | 2026-10-14 |
| Rama | exp068_compact_r7_receta (desde exp066_compact_f0) |
| Estado | run (screen fold 0) |

Idea: el blend público R7 + Raptor (LB 0.957) supera al de r6 + Raptor (0.954). El trainer de R7 es público (fuentes
embebidas en el kernel `heliosli/0-957-latest-dataset-score-in-one-click`, sha256 = manifest R7). Probamos si la parte
reproducible de la receta R7 mejora nuestra compact.

## 1. SPEC (antes de tocar el notebook)

**Receta R7 vs r6 (manifests públicos `rsna-knee-compact-r7cnxfsss` vs `-r6cnxfsss`):** mismo backbone (ConvNeXt-S
in22k-ft-in1k), 4 folds hash, 16 épocas, bs 2, 176 ventanas, crop 352/384, lr 1e-4/1e-3, wd 0.05, phys 3.5, v4. Cambia:
trainer v9.8 (sha 04df7e2e…, r6 14767c7a…); targets `opusv11b_rr7_plr7` + pesos por celda `plr7_v7` (privados); cabeza de
estados `state_weight 0.5` + `pair_rank 0.5` + `rare_boost` + `extent 0.2` (necesitan estados privados); `fm_aux` (tarea
auxiliar de cajas de lesión fastMRI+, peso 0.3: datos externos que no tenemos); `include_review_holdout`; caché
`px384_v4_sidefix`; y flags genéricos: `ema 0.999`, `slice_jitter 0.25`, `mirror 0.5`, `aug v2` (+`aug_grid_fp32`),
`train_sampling cover` (stride 3), `draw_seed`, `syn_decomp`, `compile`. En inferencia R7 añade stackers de Effusion
(volumen de líquido) y de estados (no aplican aquí).

**Hipótesis:** los flags genéricos de R7 (regularización/aumento + EMA) suben la pseudo L2 de la compact en fold 0 ≥ +0.005
sobre exp066.

**Única variable:** trainer R7 v9.8 + sus flags reproducibles (`--ema 0.999 --slice-jitter 0.25 --mirror 0.5 --aug v2
--aug-grid-fp32 --draw-seed --train-sampling cover --cover-stride 3 --syn-decomp --compile`), como paquete "receta R7
pública". Todo lo demás igual a exp066 (datos folds 1–4, etiquetas consenso como `lr_prior`, caché, split, 16 épocas,
última época, `--grad-ckpt`). Lo no reproducible queda fuera y se declara: estados/pair-rank/rare-boost/extent, pesos de
celda, fm-aux, review holdout, sidefix.

**Criterios (fijos):** compact sola fold 0 contra exp066 (pseudo 0.8755, gold 0.8972).
- `worked`: pseudo ≥ **0.8805** (+0.005) y gold ≥ **0.8772**.
- `killed`: gold < 0.8772, o GPU-h > 21.
- `didnt`: otro caso. El blend rank 0.5/0.5 con exp045 SWA f0 se reporta (no decide).
- Si `worked` → siguiente: all-data con esta receta (SPEC aparte) y candidato a sustituir exp067 en el blend Kaggle.

**Presupuesto:** ≈ 14 GPU-h RTX PRO 4500 (exp066 13.74 h; compile ~1.4× más rápido; cover/EMA casi sin coste) + caché.

**Licencia:** código heliosli solo en `public_assets` / disco del pod; nunca a git.

## 2. PLAN

- Notebook = el de exp066 con cambios en: celda 0 (EXP/PADRE, `SRC_DIR=/root/compact/src_r7`, `REF_*_COMPACT`, `R7_FLAGS`,
  `COMPILE=True`), celda 1 (sha256 R7), celda 3 (port del trainer R7, mismo cambio de folds), celda 4 (flags), celda 5
  (`Net(..., syn_decomp=True)`, decisión contra exp066), celda 6 (desvíos).
- Pod `rsna-c68` (RTX PRO 4500, 400 GB): caché folds 1–4 → smoke → train → borra caché train → caché fold 0 + gold → eval →
  commit local `metrics.json` → podStop. Worker `orq/compact2/worker_compact.sh` (DELETE_TRAIN=1).

## 3. VERDICT

(pendiente)
