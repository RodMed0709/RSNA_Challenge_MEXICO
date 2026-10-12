# sub045_4_fold_soup

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial (ensamble de 5 folds swa_f0..f4) |
| Fecha | 2026-10-12 |
| Rama | sub045_4_fold_soup |
| Estado | SPEC confirmada (Rodrigo aprobó 12-oct ~18:05 UTC) · en curso |

Idea: A5_paper_y_backlog.md ★1 (Wortsman 2022, model soups). Un solo modelo = media uniforme de los pesos de los 5 SWA
de exp045, con BN recalculada (800 estudios no-gold mezclados de todos los folds, mismo procedimiento que sub045_3).
Coste de inferencia de 1 modelo en vez de 5.

## 1. SPEC (antes de correr)

**Hipótesis:** el soup uniforme no colapsa (comparte cuenca) y en LB queda entre 1 fold y el ensamble de 5 folds.

**Única variable:** promedio de pesos (soup) en vez de promedio de predicciones.

**Evaluación:** no hay OOF limpio (el soup vio el val de cada fold a través de los otros 4 modelos).
- pseudo L2 en val fold 0: **solo informativa (leaky)**.
- gold (58): solo veto. `killed` si gold < 0.9002 (gold de 1 fold, swa_f0 0.9201, − 0.02) → soup colapsado, no se envía.
- La decisión real es el LB: 1 envío "exp045 fold soup" (sugerido 16-oct). worked si LB ≥ LB de exp045 1 fold; si no didnt.
- Se reporta también el soup sin recalcular BN (informativo).

## 2. PLAN
Notebook `sub045_4_fold_soup.ipynb` (reusa las celdas de exp045 y la función de BN de sub045_3); pesos en
`results/sub045_4_fold_soup/soup_exp045.pt` del worktree del volumen.

## 3. VERDICT
(pendiente: gold de metrics.json + LB)
