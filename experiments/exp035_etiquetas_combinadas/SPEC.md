# exp035_etiquetas_combinadas

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp021_etiquetas_L2 |
| Fecha | 2026-10-11 |
| Rama | exp035_etiquetas_combinadas |
| Estado | running |

Idea: receta exp021 (DINOv2-S, cache v1_256, 8 épocas) entrenada con el **promedio por celda** de tres fuentes blandas
públicas (L2, L3, L4) en lugar de solo L2.

Origen: `docs/pasos/paso6_plan_lb.md` (rama exp010_dev_informes). exp021 mostró que las etiquetas son la palanca del LB
(v1 → L2: +0.024). Los notebooks líderes (receta Raptor) entrenan con etiquetas blandas de varios LLM promediadas: promediar
fuentes independientes reduce el ruido de cada una.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** el promedio de varias fuentes blandas es una etiqueta menos ruidosa que una sola fuente y sube el LB público.

**Única variable cambiada:** las etiquetas de **entrenamiento**.

| Parámetro o celda | Padre (exp021) | Este experimento |
|---|---|---|
| Etiquetas de entrenamiento | L2 (`stevenleehans/rsna-knee-llm-report-labels`, `llm_labels_v4_blend.csv`) | promedio por celda de L2, L3 y L4 |

Fuentes (en el volumen, `/workspace/data/labels/public/`, descargadas por exp011; formato comprobado 2026-10-11):

| Id | Dataset / archivo | Valores | Cobertura pool (4,348) | Gold (58) |
|---|---|---|---|---|
| L2 | `stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` | blandas en [0.005, 1], 211 valores distintos | 4,348 | 58 |
| L3 | `pilkwang/rsna-knee-llm-labels/report_labels_v2.csv` | blandas, 5 niveles {0.08, 0.28, 0.68, 0.82, 0.94} (columnas `<label>`; `__conf`/`__verdict` no se usan) | 4,348 | 57 |
| L4 | `lixin73/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv` | blandas en [0, 1], 32 valores distintos | 4,348 | 58 |

Ninguna fuente es dura 0/1: las tres entran tal cual como probabilidad en [0, 1] (no se recalibran). Regla: por estudio y
hallazgo, media de las fuentes que traen valor (NaN se ignora); si ninguna trae valor, el estudio usa v1 (contado en
`N_TRAIN_FALLBACK_V1`). Comprobado antes de correr: las 52,176 celdas del pool tienen las 3 fuentes, 0 caen a v1. El
promedio entra a la BCE como objetivo blando, igual que L2 en el padre.

Todo lo demás igual a exp021: split `split_v1_seed42`, semilla, cache v1_256, DINOv2-small, 8 épocas, lr, cabeza,
`SELECT_ON = "pseudo"`, y el checkpoint se sigue eligiendo con la pseudo **contra v1** del fold 0 (celda 4 sin tocar).

**Validación:** la pseudo del fold 0 se calcula tres veces sobre el mismo checkpoint (> 0.5 → 1, < 0.5 → 0, = 0.5 fuera):
- contra v1 (comparable con exp002/exp021);
- contra L2 binarizada (comparable con exp021: 0.817);
- contra el promedio binarizado (el padre también se mide contra él, desde su `oof_f0.csv`).
Gold (58, fold −1) igual que siempre.

**Efecto esperado:** pseudo contra L2 parecida a exp021 (±0.01); LB público fold 0 sube ≥ +0.01.

**Criterios (no cambian después de correr).** Como en exp021, cambiar las etiquetas cambia la vara de la pseudo, así que
**decide el LB público de Kaggle del fold 0**, contra exp021 fold 0 (0.836):
- `worked`: LB ≥ 0.846 (+0.01).
- `didnt`: 0.826 ≤ LB < 0.846.
- `killed`: LB < 0.826, o gold < exp021 − 0.02 (fold 0 contra fold 0: gold < 0.792; exp021 gold = 0.812), o GPU-h > 1.5 ×
  presupuesto.
- Las tres pseudo se reportan, **no deciden**. La `decision_auto` del notebook (Δ pseudo v1) es informativa.

**Presupuesto:** ~3.5 GPU-h (fold 0; exp021 gastó 3.17 con GPU compartida). Corre en el pod rsna-e compartiendo GPU con
exp036: el reloj no es comparable. El envío a Kaggle (kernel de inferencia con `best_f0.pt`, plantilla de exp021) va aparte.

**Riesgos:**
- L2 (v4_blend) ya mezcla GPT-5.6-Sol y L4 es GPT-5.6-Sol: acuerdo binarizado L2–L4 = 99.2 % (L2–L3 95.4 %, L3–L4 95.0 %).
  El promedio pesa de hecho ~2/3 a GPT-5.6-Sol; el aporte nuevo real es L3 (Claude). Si el LB no se mueve, puede ser por
  esto y no porque promediar no sirva.
- L3 tiene 5 niveles y L4 media 0.26 frente a 0.34 de L2: el promedio cambia la escala de los objetivos, no el orden; no
  afecta AUC.
- L2 declara ajustes de Synovitis con los 58 gold (ver exp011): el veto gold sigue parcialmente contaminado.
- Elegir checkpoint con pseudo v1 sesga hacia v1 (igual que el padre).
- LB público ruidoso: por eso el umbral es +0.01.

## 2. PLAN

- Notebook copiado de `experiments/exp021_etiquetas_L2/exp021_etiquetas_L2.ipynb` (rama exp021_etiquetas_L2). Celdas que cambian:
  - **0** parámetros: `EXP`, `PADRE = "exp021"`, `NOTES`, `L2_CSV`/`L3_CSV`/`L4_CSV` (sustituyen a `TRAIN_LABELS_CSV`),
    `PARENT_RESULTS_DIR` → results de exp021.
  - **2** datos: `pool_train` = promedio por celda de L2, L3, L4 (fallback v1, contado en `N_TRAIN_AVG` /
    `N_TRAIN_FALLBACK_V1`; celdas por número de fuentes en `N_CELLS_BY_N_SOURCES`); `pseudo_l2_bin` y `pseudo_avg_bin`.
  - **5** evaluación: pseudo contra L2 y contra el promedio (`pseudo_auc_L2`, `pseudo_auc_AVG` y por etiqueta), también
    para el padre (`parent_pseudo_auc_L2`, `parent_pseudo_auc_AVG`), dentro de `per_fold`.
- Ninguna otra celda cambia. Smoke con `SMOKE = True`; run real con `FOLDS = [0]`.
- En el pod: symlink `results/exp021_etiquetas_L2` en el worktree para `leer_metricas(PADRE)`.
- Tras el run: kernel de inferencia en Kaggle con `best_f0.pt` (plantilla de exp021) y LB al registro de envíos.

## 3. VERDICT (números de `results/exp035_etiquetas_combinadas/metrics.json` y LB de Kaggle)

**Resultado:** · **Fase alcanzada:**

| Métrica | Padre (exp021 f0) | Este | Δ |
|---|---|---|---|
| LB público fold 0 (decide) | 0.836 | | |
| pseudo AUC fold 0 contra v1 (informa) | 0.789 | | |
| pseudo AUC fold 0 contra L2 (informa) | 0.817 | | |
| pseudo AUC fold 0 contra promedio (informa) | | | |
| gold AUC (fold 0 contra fold 0) | 0.812 | | |
| Época del mejor checkpoint | 7 de 8 | | |
| GPU-h | 3.17 | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
