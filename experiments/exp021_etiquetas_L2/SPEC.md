# exp021_etiquetas_L2

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp002_dinov2s_ref |
| Fecha | 2026-10-08 |
| Rama | exp021_etiquetas_L2 |
| Estado | spec |

Idea: receta exp002 (8 épocas, cache v1_256) entrenada con las etiquetas L2 en lugar de v1.

Origen: `docs/pasos/envios_kaggle.md` (rama exp010_dev_informes). La pseudo AUC interna sube (0.83 → 0.85) pero el LB
público se queda en 0.81 (exp002 5 folds 0.812, exp015 fold 0 0.810; mediana del LB 0.936). Las etiquetas de Kaggle siguen
definiciones oficiales estrictas (ACL/MCL solo rotura de alto grado o completa; derrame y Baker solo moderado/grande; OA con
pérdida de cartílago > 50 %) y v1 no. L2 es el candidato de exp011 con mayor acuerdo medio.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** entrenar con etiquetas alineadas a la definición oficial sube el LB público, aunque la pseudo contra v1
pueda bajar.

**Única variable cambiada:** las etiquetas de **entrenamiento**.

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| Etiquetas de entrenamiento | v1 (`report_labels_v1.csv`, Qwen2.5-7B, 0/0.5/1) | L2 (`stevenleehans/rsna-knee-llm-report-labels`, `llm_labels_v4_blend.csv`, probabilidades en [0, 1]) |

L2 en el volumen: `/workspace/data/labels/public/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv`
(descargado por exp011). Mismos 12 hallazgos y mismo manejo que v1: el valor entra tal cual a la BCE como objetivo blando
(v1 ya mete 0.5 así), NaN se enmascara. Si L2 no cubre un estudio, ese estudio usa v1 y se reporta cuántos
(comprobado antes de correr: L2 cubre los 4,348 estudios del pool, 0 caen a v1).

Todo lo demás igual: split `split_v1_seed42`, semilla, cache v1_256, DINOv2-small, 8 épocas, lr, cabeza,
`SELECT_ON = "pseudo"`. El checkpoint se sigue eligiendo con la pseudo **contra v1** del fold 0 (celda 4 sin tocar), igual
que el padre.

**Validación:** la pseudo del fold 0 se calcula dos veces sobre el mismo checkpoint:
- contra v1 (comparable con el padre; AUC solo sobre 0/1, el 0.5 queda fuera, como siempre);
- contra L2 binarizada: > 0.5 → 1, < 0.5 → 0, exactamente 0.5 → fuera (0.06 % de los valores).
Gold (58, fold −1) igual que siempre.

**Efecto esperado:** pseudo contra v1 baja o queda igual; LB público fold 0 sube ≥ +0.01.

**Criterios (no cambian después de correr). Desviación declarada del SDD:** cambiar las etiquetas cambia la vara de la
pseudo, así que la pseudo no puede decidir. **Decide el LB público de Kaggle del fold 0**, contra exp015 fold 0 (0.810) y
exp002 (0.812):
- `worked`: LB ≥ 0.822 (+0.01).
- `didnt`: LB dentro de ±0.01 (0.802 ≤ LB < 0.822).
- `killed`: LB < 0.802, o gold < exp002 − 0.02 (fold 0 contra fold 0: gold < 0.757; exp002 fold 0 gold = 0.777), o GPU-h >
  1.5 × presupuesto.
- La pseudo contra v1 y contra L2 se reportan, **no deciden**. La `decision_auto` del notebook (Δ pseudo v1) es informativa.

> **Rodrigo debe revisar y confirmar esta desviación** (decide el LB, no la pseudo) antes del verdict.

**Presupuesto:** ~2.5 GPU-h (fold 0, RTX PRO 4500 compartida con exp018/019/020: el tiempo de reloj no es comparable).
El envío a Kaggle (kernel de inferencia con `best_f0.pt`) va aparte y no cuenta como GPU-h de entrenamiento.

**Riesgos:**
- L2 declara validación y ajustes de Synovitis sobre los 58 gold (ver exp011): el veto gold queda contaminado, sobre todo en
  Synovitis. Se reporta gold por etiqueta.
- Elegir checkpoint con pseudo v1 sesga hacia v1; con 8 épocas y coseno el mejor suele ser la última, efecto pequeño.
- El LB público es ruidoso (pocos estudios públicos): ±0.01 puede ser azar. Por eso el umbral es +0.01 y no +0.005.
- Objetivos blandos de L2 (0.25 en muchos "no mencionado") empujan las predicciones hacia arriba: no afecta AUC.

## 2. PLAN

- Notebook copiado de `experiments/exp002_dinov2s_ref/exp002_dinov2s_ref.ipynb` (main). Celdas que cambian:
  - **0** parámetros: `EXP`, `PADRE = "exp002"`, `NOTES`, `TRAIN_LABELS_CSV` (ruta de L2). `LABELS_CSV` sigue en v1.
  - **2** datos: `pool_train` = pool con etiquetas L2 (fallback v1 por estudio sin L2 completo, contado en
    `N_TRAIN_L2` / `N_TRAIN_FALLBACK_V1`, que quedan en `params`); el loader de entrenamiento usa `pool_train`, validación y
    gold sin cambios; `pseudo_l2_bin` para la celda 5.
  - **5** evaluación: añade la pseudo contra L2 binarizada por fold (`per_fold[f]["pseudo_auc_L2"]` y por etiqueta) y la
    deja dentro de `per_fold` para que la celda 6 la guarde sin cambios.
- Ninguna otra celda cambia. Smoke con `SMOKE = True` antes del run real con `FOLDS = [0]`.
- Tras el run: kernel de inferencia en Kaggle con `best_f0.pt` (misma plantilla que exp015) y LB al registro de envíos.

## 3. VERDICT (números de `results/exp021_etiquetas_L2/metrics.json` y LB de Kaggle)

**Resultado:** · **Fase alcanzada:**

| Métrica | Padre (exp002 f0) | Este | Δ |
|---|---|---|---|
| LB público fold 0 (decide) | 0.812 (5 folds) / exp015 f0 0.810 | | |
| pseudo AUC fold 0 contra v1 (informa) | 0.827 | | |
| pseudo AUC fold 0 contra L2 (informa) | | | |
| gold AUC (fold 0 contra fold 0) | 0.777 | | |
| Época del mejor checkpoint | 8 de 8 | | |
| GPU-h | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
