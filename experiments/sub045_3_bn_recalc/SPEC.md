# sub045_3_bn_recalc

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial (swa_f0.pt) |
| Fecha | 2026-10-12 |
| Rama | sub045_3_bn_recalc |
| Estado | SPEC confirmada (Rodrigo aprobó 12-oct ~18:05 UTC) · en curso |

Idea: el "SWA" de la receta promedia los pesos de las 3 mejores épocas (`average_states`) y también promedia
`running_mean`/`running_var` de BatchNorm sin recalcularlos. Verificado en el código (celda 10 de exp045: no hay
`update_bn` ni pasada de BN) y en timm: `coatnet_rmlp_2_rw_384` tiene **25 capas BatchNorm** (stem + bloques MBConv de
stages 0–1). Izmailov 2018 recalcula BN tras promediar (A2_implementacion_vs_papers.md, fila "SWA top-3").

## 1. SPEC (antes de correr)

**Hipótesis:** recalcular las estadísticas de BN del SWA de fold 0 con una pasada forward sobre estudios de TRAIN del
fold sube la pseudo AUC L2 en val fold 0 en ≥ +0.002.

**Única variable:** estadísticas BN del checkpoint (promediadas → recalculadas). Pesos idénticos.

| | Padre (swa_f0 de exp045) | Este sub-experimento |
|---|---|---|
| Pesos | media de épocas [15, 14, 13] | idénticos |
| BN running stats | media de las 3 épocas | recalculadas: `reset_running_stats`, `momentum=None` (media acumulada), solo módulos BN en modo train, forward del backbone sin grad |
| Datos BN | — | 800 estudios de TRAIN del fold (muestra fija, `RandomState(0)`), loader de evaluación del notebook (K_EVAL 24 ventanas linspace, sin aug, bf16), mismo corpus raptor96_384 (leído del volumen) |
| Evaluación | `predict` + `score` de exp045 | **el mismo código** (se ejecutan las celdas 2/4/6/8/10 del notebook de exp045), sin TTA como en el padre |

**Criterio fijo:**
- `worked` si pseudo L2 f0 (SWA+BN) ≥ 0.8894655 + 0.002 = **0.8914655** (swa_f0 de `results/exp045_raptor_aug_espacial/metrics.json`: 0.8894654941741685) **y** gold ≥ 0.9201083 − 0.02 = 0.9001083.
- `killed` si gold < 0.9001083.
- `didnt` en otro caso.
- Gold solo veta. Sanidad: re-evaluar swa_f0 sin cambios en gold debe dar 0.92011 (± 1e-3).

**Si worked:** se producen `swa_bn_fK.pt` (K = 1..4, junto a `wt_exp045_fK/results/exp045_raptor_aug_espacial/swa_fK.pt`)
y `swa_bn_alldata.pt` (junto a `swa_alldata.pt`, BN con 800 estudios no-gold), con gold y pseudo de su fold informativos.

## 2. PLAN
Notebook `sub045_3_bn_recalc.ipynb`, corre en un pod GPU con auto-stop (worker `/workspace/users/rodrigo/p7c/`).
Solo inferencia (~0.3 GPU-h f0; ~1 GPU-h con folds 1–4 + all-data).

## 3. VERDICT (números de `results/sub045_3_bn_recalc/metrics.json`)

**Verdict: `didnt`** (orquestador en la nube, 2026-10-12 21:15 UTC, criterios fijos de la sección 1).
- pseudo L2 f0 SWA + BN recalculada **0.8907** vs swa_f0 0.8895: Δ **+0.0012**
  < +0.002 (umbral worked 0.8915).
- gold 0.9217 (Δ +0.0016; veto < 0.9001 no se activa). v1 0.8437.
- Sanidad: gold de swa_f0 sin tocar = 0.92011 (esperado 0.92011) ✔.
- Lectura: recalcular BN tras SWA mejora un poco pero no llega al umbral; no se generan swa_bn_f1..f4 ni all-data y no
  hay envío. Solo queda `swa_bn_f0.pt`.
