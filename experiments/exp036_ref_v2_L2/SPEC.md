# exp036_ref_v2_L2

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp012_entrada_v2 |
| Fecha | 2026-10-11 |
| Rama | exp036_ref_v2_L2 |
| Estado | running |

Idea: **referencia combinada**. Receta exp012 (cache v2_05mm, 336 px, 6 slots, DINOv2-S) entrenada con etiquetas L2 y 12
épocas.

Origen: `docs/pasos/paso6_plan_lb.md` y siguiente paso sugerido en el verdict de exp021 (referencia nueva = entrada v2 +
más épocas + etiquetas L2).

## 1. SPEC (antes de tocar el notebook)

**Desviación declarada del SDD (una variable por experimento):** este experimento **combina tres variables ya validadas por
separado**, para fijar la nueva referencia, no para medir una variable nueva:
- entrada v2 (exp012: +0.023 pseudo v1, +0.007 gold sobre exp002) — ya está en el padre;
- etiquetas de entrenamiento L2 (exp021: LB +0.024 sobre exp002);
- más épocas (exp015: 16 épocas, +0.020 pseudo y gold, meseta desde la 8, mejor la 12; exp020: ganancias se suman con la
  cabeza). Se usan 12 y no 16 por costo: v2 cuesta ~2× por época y 16 épocas en v2 pasarían de 7 GPU-h.

No se atribuye el resultado a ninguna de las dos variables cambiadas: es una referencia.

**Hipótesis:** las ganancias de v2, L2 y más épocas se suman.

**Variables cambiadas contra exp012:**

| Parámetro o celda | Padre (exp012) | Este experimento |
|---|---|---|
| Etiquetas de entrenamiento | v1 (`report_labels_v1.csv`) | L2 (`stevenleehans/rsna-knee-llm-report-labels`, `llm_labels_v4_blend.csv`), igual que exp021 |
| `EPOCHS` | 8 | 12 |

L2 se maneja exactamente como en exp021: valor tal cual a la BCE como objetivo blando, NaN enmascarado, fallback v1 por
estudio sin L2 completo (`N_TRAIN_L2` / `N_TRAIN_FALLBACK_V1`; L2 cubre los 4,348 estudios del pool v1, se espera 0 fallback
también en el pool v2). Todo lo demás igual a exp012: split `split_v1_seed42`, semilla, cache v2_05mm, 336 px, 6 slots,
batch 1 × acumulación 8, lr, cabeza, `FREEZE_EPOCHS = 1`, coseno con warmup 10 % (se estira a 12 épocas),
`SELECT_ON = "pseudo"`, checkpoint elegido con la pseudo **contra v1** del fold 0 (celda 4 sin tocar, igual que exp021).

**Validación:** pseudo del fold 0 sobre el mismo checkpoint contra v1 y contra L2 binarizada (> 0.5 → 1, < 0.5 → 0,
= 0.5 fuera), como exp021; también la del padre (exp012) contra L2 desde su `oof_f0.csv`. Gold (58, fold −1) igual.

**Efecto esperado:** pseudo contra L2 fold 0 ≈ 0.83–0.85 (exp021 0.817 + entrada v2 + épocas).

**Criterios (no cambian después de correr).** La referencia a batir es exp021 (misma vara L2):
- `worked`: pseudo contra L2 fold 0 ≥ 0.827 (exp021 0.817 + 0.01) **y** gold ≥ 0.792 (exp021 fold 0 = 0.812, no cae > 0.02).
- `killed`: gold < 0.792, o GPU-h > 1.5 × presupuesto.
- `didnt`: el resto (pseudo contra L2 < 0.827 sin veto).
- La pseudo contra v1 y la `decision_auto` (Δ pseudo v1 contra exp012) se reportan, no deciden.
- El LB público del fold 0 se mide después (kernel de inferencia v2) y se reporta; no cambia este criterio.

**Presupuesto:** ~6 GPU-h (fold 0; exp012 gastó 3.65 en 8 épocas con GPU compartida → × 12/8 ≈ 5.5). Corre en el pod
rsna-e compartiendo GPU con exp035.

**Riesgos:**
- Combinar variables impide atribuir; aceptado por ser referencia (ver desviación).
- L2 declara ajustes de Synovitis con los 58 gold: el veto gold está parcialmente contaminado.
- Elegir checkpoint con pseudo v1 sesga hacia v1; con 12 épocas y coseno el mejor suele ser una de las últimas.
- El envío a Kaggle necesita replicar el preprocesado v2 en el kernel de inferencia (pendiente, fuera de este screen).
- Memoria GPU compartida con exp035: si no cabe, se baja batch y se sube acumulación (no cuenta como variable).

## 2. PLAN

- Notebook copiado de `experiments/exp012_entrada_v2/exp012_entrada_v2.ipynb` (rama exp012_entrada_v2). Celdas que cambian:
  - **0** parámetros: `EXP`, `NOTES`, `EPOCHS = 12`, `TRAIN_LABELS_CSV` (L2), `PARENT_RESULTS_DIR` (results de exp012).
    `PADRE = "exp012_entrada_v2"` y `LABELS_CSV` (v1, validación) sin cambio.
  - **2** datos: `pool_train` con L2 (código de exp021) para el loader de entrenamiento; `pseudo_l2_bin`.
  - **5** evaluación: pseudo contra L2 del modelo y del padre (código de exp021), dentro de `per_fold`.
- Ninguna otra celda cambia. Smoke con `SMOKE = True`; run real con `FOLDS = [0]`.
- En el pod: symlink `results/exp012_entrada_v2` en el worktree para `leer_metricas(PADRE)`.

## 3. VERDICT (números de `results/exp036_ref_v2_L2/metrics.json`)

**Resultado:** · **Fase alcanzada:**

| Métrica | exp012 f0 (padre) | exp021 f0 (vara) | Este | Δ vs exp021 |
|---|---|---|---|---|
| pseudo AUC fold 0 contra L2 (decide) | | 0.817 | | |
| pseudo AUC fold 0 contra v1 (informa) | 0.850 | 0.789 | | |
| gold AUC (fold 0) | 0.784 | 0.812 | | |
| LB público fold 0 (informa, después) | — | 0.836 | | |
| Época del mejor checkpoint | 8 de 8 | 7 de 8 | | |
| GPU-h | 3.65 | 3.17 | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
